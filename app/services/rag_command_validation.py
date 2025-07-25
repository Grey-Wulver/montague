# File: ~/net-chatbot/app/services/rag_command_validation.py
# RAG Command Validation Service - Barrow Integration for Smart Command Suggestions

"""
RAG Command Validation Service - Phase 6 Smart Command Suggestions

Integrates with Barrow RAG system (port 8001) to provide:
- Command validation using 40k+ vendor documentation chunks
- Smart command suggestions for invalid commands
- Documentation-backed explanations for corrections
- Learning integration to improve future suggestions

Architecture Integration:
User Input → LLM Intent Parser → Three-Tier Discovery →
                                    ↓
[NEW] RAG Command Validation → Command Execution → Universal Formatter → Response

Maintains VS Code + Ruff + Black standards with 88-character line length.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class CommandValidationResult:
    """Result from RAG command validation"""

    command: str
    is_valid: bool
    confidence: float

    # For invalid commands
    suggested_command: Optional[str] = None
    explanation: Optional[str] = None
    documentation_source: Optional[str] = None

    # Performance metrics
    validation_time: float = 0.0
    rag_response_time: float = 0.0

    # Error handling
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass
class CommandCorrectionSuggestion:
    """Command correction suggestion for user interaction"""

    original_command: str
    suggested_command: str
    vendor: str
    platform: str
    confidence: float
    explanation: str
    documentation_reference: str

    # User interaction
    requires_confirmation: bool = True
    auto_apply_threshold: float = 0.95  # Auto-apply if confidence > 95%


class RAGCommandValidationService:
    """
    RAG-powered command validation service for enterprise-grade command suggestions.

    Features:
    - Pre-execution command validation
    - Documentation-backed command suggestions
    - Interactive correction workflow
    - Learning integration with existing PostgreSQL
    - Graceful fallback when RAG unavailable
    """

    def __init__(
        self,
        barrow_url: str = "http://localhost:8001",
        timeout: int = 10,
        max_retries: int = 2,
        confidence_threshold: float = 0.8,
    ):
        self.barrow_url = barrow_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

        # Performance tracking
        self.stats = {
            "total_validations": 0,
            "valid_commands": 0,
            "invalid_commands": 0,
            "suggestions_provided": 0,
            "rag_service_errors": 0,
            "average_validation_time": 0.0,
            "correction_acceptance_rate": 0.0,
        }

    async def health_check(self) -> bool:
        """Check if Barrow RAG service is available"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{self.barrow_url}/api/v1/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"RAG service health check failed: {e}")
            return False

    async def validate_command(
        self,
        command: str,
        vendor: str,
        platform: str,
        device_name: str = None,
    ) -> CommandValidationResult:
        """
        Validate command using RAG documentation knowledge.

        Returns validation result with suggestions if command is invalid.
        """
        start_time = time.time()
        self.stats["total_validations"] += 1

        logger.info(f"🔍 RAG validating command: '{command}' for {vendor} {platform}")

        try:
            # Build RAG query for command validation
            validation_query = self._build_validation_query(command, vendor, platform)

            # Call Barrow RAG service
            rag_result = await self._query_barrow_rag(validation_query, vendor)

            if not rag_result:
                return self._fallback_validation(command, vendor, platform, start_time)

            # Parse RAG response for validation
            validation_result = await self._parse_validation_response(
                rag_result, command, vendor, platform, start_time
            )

            # Update statistics
            if validation_result.is_valid:
                self.stats["valid_commands"] += 1
            else:
                self.stats["invalid_commands"] += 1
                if validation_result.suggested_command:
                    self.stats["suggestions_provided"] += 1

            self._update_performance_stats(validation_result.validation_time)

            logger.info(
                f"✅ RAG validation completed: Valid={validation_result.is_valid}, "
                f"Confidence={validation_result.confidence:.2f}, "
                f"Time={validation_result.validation_time:.2f}s"
            )

            return validation_result

        except Exception as e:
            self.stats["rag_service_errors"] += 1
            logger.error(f"❌ RAG validation failed: {e}")
            return self._fallback_validation(
                command, vendor, platform, start_time, str(e)
            )

    async def validate_discovered_commands(
        self,
        discovery_results: Dict[str, Any],  # DiscoveryInfo objects
        devices: List[Any],  # Device objects
    ) -> Dict[str, CommandValidationResult]:
        """
        Validate all discovered commands before execution.

        This is the main integration point with Universal Request Processor.
        """
        validation_results = {}

        logger.info(f"🔍 RAG validating {len(discovery_results)} discovered commands")

        # Create device lookup for efficient access
        device_lookup = {device.name: device for device in devices}

        # Validate each discovered command
        validation_tasks = []
        for device_name, discovery_info in discovery_results.items():
            device = device_lookup.get(device_name)
            if device:
                task = self.validate_command(
                    command=discovery_info.discovered_command,
                    vendor=device.vendor,
                    platform=device.platform,
                    device_name=device_name,
                )
                validation_tasks.append((device_name, task))

        # Execute validations concurrently
        for device_name, task in validation_tasks:
            try:
                validation_result = await task
                validation_results[device_name] = validation_result
            except Exception as e:
                logger.error(f"Validation failed for {device_name}: {e}")
                # Continue with other validations

        return validation_results

    def generate_correction_suggestions(
        self,
        validation_results: Dict[str, CommandValidationResult],
        discovery_results: Dict[str, Any],
        devices: List[Any],
    ) -> List[CommandCorrectionSuggestion]:
        """
        Generate user-friendly correction suggestions for invalid commands.
        """
        suggestions = []
        device_lookup = {device.name: device for device in devices}

        for device_name, validation_result in validation_results.items():
            if not validation_result.is_valid and validation_result.suggested_command:
                device = device_lookup.get(device_name)
                discovery_info = discovery_results.get(device_name)

                if device and discovery_info:
                    suggestion = CommandCorrectionSuggestion(
                        original_command=discovery_info.discovered_command,
                        suggested_command=validation_result.suggested_command,
                        vendor=device.vendor,
                        platform=device.platform,
                        confidence=validation_result.confidence,
                        explanation=validation_result.explanation
                        or "Command syntax correction",
                        documentation_reference=validation_result.documentation_source
                        or "Vendor documentation",
                    )
                    suggestions.append(suggestion)

        return suggestions

    def _build_validation_query(self, command: str, vendor: str, platform: str) -> str:
        """Build RAG query for command validation"""

        # Create specific validation query for Barrow
        query = f"""
        Validate this network command syntax for {vendor} {platform}:

        Command: {command}

        Is this command valid? If not, what is the correct syntax?
        Provide the exact correct command if available.
        """

        return query.strip()

    async def _query_barrow_rag(
        self, query: str, vendor: str = None
    ) -> Optional[Dict[str, Any]]:
        """Query Barrow RAG service for command validation"""

        rag_start_time = time.time()

        # Build RAG API request
        request_payload = {
            "question": query,
            "vendor": vendor.lower() if vendor else None,
            "context_limit": 3,
            "include_sources": True,
        }

        for attempt in range(self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.barrow_url}/api/v1/rag/query",
                        json=request_payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            rag_time = time.time() - rag_start_time

                            logger.debug(
                                f"RAG query successful in {rag_time:.2f}s: "
                                f"Confidence={result.get('confidence', 0):.2f}"
                            )

                            return result
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"RAG service returned {response.status}: {error_text}"
                            )

            except Exception as e:
                logger.warning(f"RAG query attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        return None

    async def _parse_validation_response(
        self,
        rag_result: Dict[str, Any],
        original_command: str,
        vendor: str,
        platform: str,
        start_time: float,
    ) -> CommandValidationResult:
        """Parse RAG response to determine command validity and suggestions"""

        validation_time = time.time() - start_time

        # Extract key information from RAG response
        answer = rag_result.get("answer", "").lower()
        confidence = rag_result.get("confidence", 0.0)
        sources = rag_result.get("sources", [])

        # Determine if command is valid based on RAG response
        is_valid = self._analyze_command_validity(answer, confidence)

        # Extract suggested command if invalid
        suggested_command = None
        explanation = None
        documentation_source = None

        if not is_valid:
            suggested_command = self._extract_suggested_command(
                answer, original_command
            )
            explanation = self._extract_explanation(answer)
            documentation_source = self._extract_documentation_source(sources)

        return CommandValidationResult(
            command=original_command,
            is_valid=is_valid,
            confidence=confidence,
            suggested_command=suggested_command,
            explanation=explanation,
            documentation_source=documentation_source,
            validation_time=validation_time,
            rag_response_time=rag_result.get("processing_time", 0.0),
        )

    def _analyze_command_validity(self, answer: str, confidence: float) -> bool:
        """Analyze RAG answer to determine if command is valid"""

        # Low confidence suggests uncertainty
        if confidence < self.confidence_threshold:
            return False

        # Look for validity indicators in the answer
        invalid_indicators = [
            "invalid",
            "incorrect",
            "wrong",
            "not valid",
            "does not exist",
            "not supported",
            "syntax error",
            "command not found",
            "should be",
            "correct command",
            "instead use",
        ]

        valid_indicators = [
            "valid command",
            "correct syntax",
            "proper command",
            "exists",
            "supported",
            "available",
        ]

        answer_lower = answer.lower()

        # Check for explicit invalid indicators
        for indicator in invalid_indicators:
            if indicator in answer_lower:
                return False

        # Check for explicit valid indicators
        for indicator in valid_indicators:
            if indicator in answer_lower:
                return True

        # Default to valid if high confidence and no invalid indicators
        return confidence > 0.9

    def _extract_suggested_command(
        self, answer: str, original_command: str
    ) -> Optional[str]:
        """Extract suggested command from RAG answer"""

        # Look for common suggestion patterns
        suggestion_patterns = [
            r"correct command is:?\s*([^\n\r.]+)",
            r"should be:?\s*([^\n\r.]+)",
            r"use:?\s*([^\n\r.]+)",
            r"instead:?\s*([^\n\r.]+)",
            r"`([^`]+)`",  # Commands in backticks
        ]

        import re

        for pattern in suggestion_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                suggested = match.group(1).strip()
                # Clean up the suggestion
                suggested = re.sub(r'^["\']|["\']$', "", suggested)  # Remove quotes
                suggested = suggested.split(".")[0]  # Take first sentence

                if suggested and suggested != original_command:
                    return suggested

        return None

    def _extract_explanation(self, answer: str) -> Optional[str]:
        """Extract explanation from RAG answer"""

        # Take first sentence as explanation, limit length
        sentences = answer.split(".")
        if sentences:
            explanation = sentences[0].strip()
            if len(explanation) > 200:
                explanation = explanation[:200] + "..."
            return explanation

        return None

    def _extract_documentation_source(self, sources: List[Dict]) -> Optional[str]:
        """Extract documentation source reference"""

        if sources:
            # Use first source as primary reference
            source = sources[0]
            source_name = source.get("source", "")
            if source_name:
                return f"Source: {source_name}"

        return "Vendor documentation"

    def _fallback_validation(
        self,
        command: str,
        vendor: str,
        platform: str,
        start_time: float,
        error_message: str = None,
    ) -> CommandValidationResult:
        """Fallback validation when RAG service unavailable"""

        validation_time = time.time() - start_time

        logger.warning(f"Using fallback validation for: {command}")

        # Basic heuristic validation as fallback
        is_valid = self._basic_command_heuristics(command, platform)

        return CommandValidationResult(
            command=command,
            is_valid=is_valid,
            confidence=0.5,  # Lower confidence for fallback
            validation_time=validation_time,
            error_message=error_message,
            fallback_used=True,
        )

    def _basic_command_heuristics(self, command: str, platform: str) -> bool:
        """Basic command validation heuristics as fallback"""

        # Very basic validation - just check if it looks like a show command
        command_lower = command.lower().strip()

        # Must start with 'show' for safety
        if not command_lower.startswith("show"):
            return False

        # Check for obviously invalid patterns
        invalid_patterns = [
            "show interface state",  # Should be 'status'
            "show int state",
            "show interfaces state",
        ]

        for pattern in invalid_patterns:
            if pattern in command_lower:
                return False

        # Default to valid for show commands
        return True

    def _update_performance_stats(self, validation_time: float):
        """Update performance statistics"""

        # Update average validation time
        total_validations = self.stats["total_validations"]
        current_avg = self.stats["average_validation_time"]

        self.stats["average_validation_time"] = (
            current_avg * (total_validations - 1) + validation_time
        ) / total_validations


# Global service instance
rag_validation_service = None


async def create_rag_validation_service() -> RAGCommandValidationService:
    """Factory function to create RAG validation service"""
    global rag_validation_service

    if not rag_validation_service:
        rag_validation_service = RAGCommandValidationService()

        # Check if RAG service is available
        service_available = await rag_validation_service.health_check()
        if service_available:
            logger.info("✅ RAG Command Validation Service initialized successfully")
        else:
            logger.warning("⚠️ RAG service unavailable - will use fallback validation")

    return rag_validation_service


async def get_rag_validation_service() -> RAGCommandValidationService:
    """Get initialized RAG validation service"""
    return await create_rag_validation_service()
