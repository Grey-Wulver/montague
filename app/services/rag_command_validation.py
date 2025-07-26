# app/services/rag_command_validation.py
"""
RAG Command Validation Service - Barrow Integration for Smart Command Suggestions

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
import os
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


class RAGCommandValidationService:
    """
    RAG-powered command validation service for enterprise-grade command suggestions.

    Integrates with Barrow RAG service to validate commands against 40k+ documentation chunks.
    Provides smart suggestions for invalid commands with vendor documentation backing.
    """

    def __init__(self):
        """Initialize RAG validation service with Barrow integration"""

        # FIXED: Use environment variable with correct URL
        self.rag_service_url = os.getenv(
            "RAG_SERVICE_URL", "http://192.168.1.11:8001/api/v1"
        )
        self.rag_timeout = 10
        self.max_retries = 2

        # Validation configuration
        self.validation_confidence_threshold = 0.8
        self.suggestion_confidence_threshold = 0.7
        self.auto_apply_threshold = 0.95

        # Performance tracking
        self.stats = {
            "total_validations": 0,
            "valid_commands": 0,
            "invalid_commands": 0,
            "suggestions_provided": 0,
            "rag_service_errors": 0,
            "fallback_validations": 0,
            "average_validation_time": 0.0,
        }

    async def validate_command(
        self,
        command: str,
        vendor: str,
        platform: str,
        device_context: Optional[Dict] = None,
    ) -> CommandValidationResult:
        """
        Validate a single command using RAG service.

        Args:
            command: The command to validate
            vendor: Device vendor (arista, cisco, juniper, etc.)
            platform: Platform (arista_eos, cisco_ios, etc.)
            device_context: Optional device context for validation

        Returns:
            CommandValidationResult with validation outcome and suggestions
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
        logger.info(f"🔍 RAG validating {len(discovery_results)} discovered commands")

        validation_results = {}

        # Validate each discovered command
        for device_name, discovery_info in discovery_results.items():
            # Find the corresponding device object
            device = next((d for d in devices if d.name == device_name), None)
            if not device:
                logger.warning(f"Device {device_name} not found for validation")
                continue

            # Validate the discovered command
            validation_result = await self.validate_command(
                command=discovery_info.discovered_command,
                vendor=device.vendor,
                platform=device.platform,
                device_context={"device_name": device_name, "site": device.site},
            )

            validation_results[device_name] = validation_result

        return validation_results

    def _build_validation_query(self, command: str, vendor: str, platform: str) -> str:
        """Build RAG query for command validation"""

        query = f"""
        Validate this network command for {vendor} {platform}:
        Command: {command}

        Is this command valid? If not, what is the correct command?
        Include explanation and reference to documentation.
        """

        return query.strip()

    async def _query_barrow_rag(
        self, query: str, vendor: str
    ) -> Optional[Dict[str, Any]]:
        """Query Barrow RAG service for command validation"""

        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(self.rag_timeout)
                ) as session:
                    # FIXED: Use correct payload format
                    payload = {
                        "question": query,  # Use 'question' not 'query'
                        "vendor": vendor,
                    }

                    # FIXED: Use correct endpoint /rag/query
                    async with session.post(
                        f"{self.rag_service_url}/rag/query", json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            logger.warning(
                                f"RAG service returned {response.status}: {await response.text()}"
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
            rag_response_time=validation_time,  # Same as validation time for now
        )

    def _analyze_command_validity(self, answer: str, confidence: float) -> bool:
        """Analyze RAG answer to determine if command is valid"""

        # Check for explicit validity indicators
        valid_indicators = ["valid", "correct", "proper", "accurate"]
        invalid_indicators = ["invalid", "incorrect", "wrong", "error", "should be"]

        valid_score = sum(1 for indicator in valid_indicators if indicator in answer)
        invalid_score = sum(
            1 for indicator in invalid_indicators if indicator in answer
        )

        # If confidence is high and no invalid indicators, consider valid
        if confidence >= self.validation_confidence_threshold and invalid_score == 0:
            return True

        # If more invalid indicators than valid, consider invalid
        if invalid_score > valid_score:
            return False

        # Default based on confidence threshold
        return confidence >= self.validation_confidence_threshold

    def _extract_suggested_command(
        self, answer: str, original_command: str
    ) -> Optional[str]:
        """Extract suggested command from RAG answer"""

        # Look for common suggestion patterns
        suggestion_patterns = [
            "should be",
            "correct command is",
            "use instead",
            "try",
        ]

        for pattern in suggestion_patterns:
            if pattern in answer:
                # Extract text after the pattern
                parts = answer.split(pattern, 1)
                if len(parts) > 1:
                    suggested = parts[1].split(".")[0].strip()  # Take first sentence

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
        self.stats["fallback_validations"] += 1

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

    async def health_check(self) -> bool:
        """Check if RAG service is available"""

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(5)
            ) as session:
                async with session.get(f"{self.rag_service_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""

        stats = self.stats.copy()

        # Calculate derived metrics
        if stats["total_validations"] > 0:
            stats["validation_success_rate"] = (
                stats["valid_commands"] / stats["total_validations"]
            )
            stats["suggestion_rate"] = (
                stats["suggestions_provided"] / stats["invalid_commands"]
                if stats["invalid_commands"] > 0
                else 0
            )

        return stats


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
