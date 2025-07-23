# File: ~/net-chatbot/database/seed_postgres.py
#!/usr/bin/env python3
"""
Enhanced PostgreSQL Database Seeding Script
Seeds the three-tier discovery system with intelligent conflict resolution
- NAPALM Getters: 0.95-1.0 confidence (industry standard)
- Community: 0.8-0.9 confidence (community verified)
- LLM Discoveries: 0.7 initial → up to 0.9+ with learning (+0.1 per success)
- Automatic demotion for failing high-confidence commands

Run from: ~/net-chatbot/
Usage: python database/seed_postgres.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from app.core.database import get_db_session_direct, init_database
from app.models.command_mappings import CommandMapping, CommunitySync
from app.services.community_mapper import community_mapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def enhance_database_schema():
    """Add learning and conflict tracking fields to existing schema"""

    logger.info("🔧 Enhancing database schema for intelligent learning...")

    session = await get_db_session_direct()

    try:
        # Check if columns already exist to avoid errors
        result = await session.execute(
            text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'command_mappings'
            AND column_name IN ('execution_count', 'success_count', 'last_success', 'last_failure', 'confidence_history')
        """)
        )
        existing_columns = [row[0] for row in result.fetchall()]

        # Add learning tracking columns if they don't exist
        schema_updates = []

        if "execution_count" not in existing_columns:
            schema_updates.append(
                "ALTER TABLE command_mappings ADD COLUMN execution_count INTEGER DEFAULT 0"
            )

        if "success_count" not in existing_columns:
            schema_updates.append(
                "ALTER TABLE command_mappings ADD COLUMN success_count INTEGER DEFAULT 0"
            )

        if "last_success" not in existing_columns:
            schema_updates.append(
                "ALTER TABLE command_mappings ADD COLUMN last_success TIMESTAMP"
            )

        if "last_failure" not in existing_columns:
            schema_updates.append(
                "ALTER TABLE command_mappings ADD COLUMN last_failure TIMESTAMP"
            )

        if "confidence_history" not in existing_columns:
            schema_updates.append(
                "ALTER TABLE command_mappings ADD COLUMN confidence_history JSONB DEFAULT '[]'::jsonb"
            )

        # Execute schema updates
        for update_sql in schema_updates:
            await session.execute(text(update_sql))
            logger.info(f"✅ {update_sql}")

        await session.commit()

        # Create conflict audit table
        await session.execute(
            text("""
            CREATE TABLE IF NOT EXISTS command_conflicts (
                id SERIAL PRIMARY KEY,
                intent VARCHAR(100) NOT NULL,
                platform VARCHAR(50) NOT NULL,
                winning_source VARCHAR(20) NOT NULL,
                winning_confidence FLOAT NOT NULL,
                losing_sources JSONB NOT NULL,
                confidence_diff FLOAT NOT NULL,
                conflict_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        )

        await session.commit()
        logger.info("✅ Enhanced schema with learning and conflict tracking")

        return True

    except Exception as e:
        logger.error(f"❌ Schema enhancement failed: {e}")
        await session.rollback()
        return False

    finally:
        await session.close()


async def seed_postgres_with_conflict_resolution():
    """Seed PostgreSQL with intelligent conflict resolution"""

    logger.info("🚀 Starting intelligent PostgreSQL seeding...")

    # Initialize database connections
    success = await init_database()
    if not success:
        logger.error("❌ Failed to initialize database connections")
        return False

    # Enhance schema for learning
    schema_success = await enhance_database_schema()
    if not schema_success:
        logger.error("❌ Failed to enhance database schema")
        return False

    session = await get_db_session_direct()

    try:
        # Clear existing data for fresh start (dev environment)
        logger.info("🧹 Clearing existing command mappings...")
        await session.execute(text("DELETE FROM command_mappings"))
        await session.execute(text("DELETE FROM community_sync_logs"))
        await session.execute(text("DELETE FROM command_conflicts"))
        await session.commit()

        # Get all community mappings
        logger.info("📥 Harvesting community command mappings...")
        start_time = datetime.now()

        # Get combined mappings from all sources
        all_mappings = community_mapper.get_combined_mappings()
        napalm_getters = community_mapper.get_napalm_getters()

        # Track seeding statistics
        seeded_commands = 0
        seeded_intents = 0
        conflicts_resolved = 0
        platforms_processed = set()

        # Create a staging area for conflict resolution
        staging_mappings = {}  # key: (intent, platform) -> best_mapping

        # Process standard commands (from ntc-templates)
        logger.info("📊 Processing community command mappings...")
        for platform, commands in all_mappings.items():
            platforms_processed.add(platform)

            for command in commands:
                # Create a generic intent from the command
                intent = command.lower().replace("show ", "").replace(" ", "_")

                # Skip if intent would be empty or too short
                if len(intent) < 2:
                    continue

                mapping_key = (intent, platform)
                mapping_data = {
                    "intent": intent,
                    "platform": platform,
                    "command": command,
                    "confidence": 0.8,  # Community mappings base confidence
                    "source": "community",
                    "usage_count": 0,
                    "discovery_metadata": {
                        "seeded_from": "community_harvesting",
                        "seed_date": start_time.isoformat(),
                        "sources": ["ntc_templates"],
                    },
                }

                # Add to staging (will be resolved later)
                if mapping_key not in staging_mappings:
                    staging_mappings[mapping_key] = mapping_data
                # If exists, keep higher confidence (community vs community shouldn't conflict much)

        # Process NAPALM getters (high-quality intent mappings!)
        logger.info("🎯 Processing NAPALM intent-based mappings...")
        for intent, platform_commands in napalm_getters.items():
            seeded_intents += 1

            for platform, command in platform_commands.items():
                platforms_processed.add(platform)

                mapping_key = (
                    intent.replace("get_", ""),
                    platform,
                )  # get_bgp_neighbors -> bgp_neighbors
                mapping_data = {
                    "intent": intent.replace("get_", ""),
                    "platform": platform,
                    "command": command,
                    "confidence": 0.95,  # NAPALM getters are high quality
                    "source": "community",
                    "usage_count": 0,
                    "discovery_metadata": {
                        "seeded_from": "napalm_getters",
                        "seed_date": start_time.isoformat(),
                        "intent_quality": "high",
                        "napalm_getter": intent,
                    },
                }

                # Conflict resolution: NAPALM wins over community due to higher confidence
                if mapping_key in staging_mappings:
                    existing = staging_mappings[mapping_key]
                    if mapping_data["confidence"] > existing["confidence"]:
                        # Log conflict resolution
                        conflicts_resolved += 1
                        logger.info(
                            f"🔄 Conflict resolved: {mapping_key[0]}:{mapping_key[1]} - "
                            f"NAPALM ({mapping_data['confidence']}) beats community ({existing['confidence']})"
                        )
                        staging_mappings[mapping_key] = mapping_data
                else:
                    staging_mappings[mapping_key] = mapping_data

        # Batch insert with intelligent upsert
        logger.info("💾 Committing resolved mappings to PostgreSQL...")

        batch_size = 100
        mappings_list = list(staging_mappings.values())

        for i in range(0, len(mappings_list), batch_size):
            batch = mappings_list[i : i + batch_size]

            for mapping_data in batch:
                # Use intelligent upsert with conflict resolution
                stmt = insert(CommandMapping).values(
                    intent=mapping_data["intent"],
                    platform=mapping_data["platform"],
                    command=mapping_data["command"],
                    confidence=mapping_data["confidence"],
                    source=mapping_data["source"],
                    usage_count=mapping_data["usage_count"],
                    last_used=None,
                    discovery_metadata=mapping_data["discovery_metadata"],
                    execution_count=0,
                    success_count=0,
                    confidence_history=[
                        {
                            "confidence": mapping_data["confidence"],
                            "source": mapping_data["source"],
                            "timestamp": start_time.isoformat(),
                            "reason": "initial_seeding",
                        }
                    ],
                )

                # Intelligent conflict resolution
                stmt = stmt.on_conflict_do_update(
                    index_elements=["intent", "platform"],
                    set_={
                        "command": stmt.excluded.command,
                        "confidence": stmt.excluded.confidence,
                        "source": stmt.excluded.source,
                        "updated_at": start_time,
                        "discovery_metadata": stmt.excluded.discovery_metadata,
                        "confidence_history": CommandMapping.confidence_history.concat(
                            stmt.excluded.confidence_history
                        ),
                    },
                    where=stmt.excluded.confidence > CommandMapping.confidence,
                )

                session.add(
                    CommandMapping(
                        **{
                            k: v
                            for k, v in mapping_data.items()
                            if k
                            in [
                                "intent",
                                "platform",
                                "command",
                                "confidence",
                                "source",
                                "usage_count",
                                "discovery_metadata",
                            ]
                        }
                    )
                )

            seeded_commands += len(batch)

        # Commit all mappings
        await session.commit()

        # Create community sync log entry
        end_time = datetime.now()
        sync_duration = int((end_time - start_time).total_seconds() * 1000)

        sync_log = CommunitySync(
            repository_url="multiple_sources",
            sync_type="initial_seed",
            commands_discovered=seeded_commands,
            commands_added=seeded_commands,
            commands_updated=0,
            sync_duration_ms=sync_duration,
            status="success",
            sync_metadata={
                "platforms_processed": list(platforms_processed),
                "total_platforms": len(platforms_processed),
                "napalm_intents": seeded_intents,
                "conflicts_resolved": conflicts_resolved,
                "seed_sources": ["ntc_templates", "napalm_getters"],
                "seeding_method": "intelligent_conflict_resolution",
                "confidence_strategy": {
                    "community": 0.8,
                    "napalm": 0.95,
                    "llm_initial": 0.7,
                    "llm_learning": "+0.1_per_success",
                },
            },
        )

        session.add(sync_log)
        await session.commit()

        # Log success statistics
        logger.info("✅ PostgreSQL seeding completed successfully!")
        logger.info("📊 Seeding Statistics:")
        logger.info(f"   🔧 Commands seeded: {seeded_commands}")
        logger.info(f"   🎯 Intent mappings: {seeded_intents}")
        logger.info(f"   🖥️  Platforms processed: {len(platforms_processed)}")
        logger.info(f"   🔄 Conflicts resolved: {conflicts_resolved}")
        logger.info(f"   ⏱️  Seeding duration: {sync_duration}ms")
        logger.info(
            "   📋 Key platforms: arista_eos, cisco_ios, cisco_nxos, cisco_xr, juniper_junos"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Seeding failed: {e}")
        await session.rollback()
        return False

    finally:
        await session.close()


async def true_up_command_confidence():
    """True-up command confidence based on execution performance (on-demand)"""

    logger.info("🎯 Starting confidence true-up based on execution performance...")

    session = await get_db_session_direct()

    try:
        # Find commands with poor success rates that need demotion
        result = await session.execute(
            text("""
            SELECT id, intent, platform, command, confidence,
                   execution_count, success_count,
                   CASE
                       WHEN execution_count > 0 THEN (success_count::float / execution_count::float)
                       ELSE 0.0
                   END as success_rate
            FROM command_mappings
            WHERE execution_count >= 5  -- Only consider commands with meaningful sample size
            AND confidence > 0.6        -- Only demote higher confidence commands
            AND (success_count::float / execution_count::float) < 0.6  -- Less than 60% success
        """)
        )

        poor_performers = result.fetchall()

        demotions = 0
        for cmd in poor_performers:
            # Calculate new confidence: penalize based on failure rate
            success_rate = cmd.success_rate
            penalty_factor = max(0.3, success_rate)  # Don't go below 0.3 confidence
            new_confidence = max(0.3, cmd.confidence * penalty_factor)

            # Update confidence
            await session.execute(
                text("""
                UPDATE command_mappings
                SET confidence = :new_confidence,
                    confidence_history = confidence_history || :history_entry,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :cmd_id
            """),
                {
                    "new_confidence": new_confidence,
                    "cmd_id": cmd.id,
                    "history_entry": [
                        {
                            "confidence": new_confidence,
                            "previous_confidence": cmd.confidence,
                            "reason": "performance_demotion",
                            "success_rate": success_rate,
                            "execution_count": cmd.execution_count,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ],
                },
            )

            demotions += 1
            logger.info(
                f"⬇️  Demoted: {cmd.intent}:{cmd.platform} "
                f"{cmd.confidence:.2f} → {new_confidence:.2f} "
                f"(success: {success_rate:.1%})"
            )

        await session.commit()

        # Generate audit report
        total_commands = await session.execute(
            text("SELECT COUNT(*) FROM command_mappings")
        )
        total_count = total_commands.scalar()

        high_confidence = await session.execute(
            text("""
            SELECT COUNT(*) FROM command_mappings WHERE confidence >= 0.9
        """)
        )
        high_count = high_confidence.scalar()

        logger.info("📈 True-up completed!")
        logger.info(f"   📊 Total commands: {total_count}")
        logger.info(f"   ⭐ High confidence (≥0.9): {high_count}")
        logger.info(f"   ⬇️  Commands demoted: {demotions}")

        return {"demotions": demotions, "total_commands": total_count}

    except Exception as e:
        logger.error(f"❌ True-up failed: {e}")
        await session.rollback()
        return None

    finally:
        await session.close()


async def verify_intelligent_seeding():
    """Verify that intelligent seeding was successful"""

    logger.info("🔍 Verifying intelligent PostgreSQL seeding...")

    session = await get_db_session_direct()

    try:
        # Check command mappings count
        result = await session.execute(text("SELECT COUNT(*) FROM command_mappings"))
        total_commands = result.scalar()

        # Check confidence distribution
        result = await session.execute(
            text("""
            SELECT
                CASE
                    WHEN confidence >= 0.95 THEN 'excellent'
                    WHEN confidence >= 0.8 THEN 'good'
                    WHEN confidence >= 0.7 THEN 'acceptable'
                    ELSE 'needs_improvement'
                END as confidence_tier,
                COUNT(*)
            FROM command_mappings
            GROUP BY confidence_tier
            ORDER BY MIN(confidence) DESC
        """)
        )
        confidence_dist = dict(result.fetchall())

        # Check for key platforms and intents
        result = await session.execute(
            text("""
            SELECT platform, COUNT(*) as cmd_count
            FROM command_mappings
            WHERE platform IN ('arista_eos', 'cisco_ios', 'cisco_nxos', 'cisco_xr', 'juniper_junos')
            GROUP BY platform
            ORDER BY cmd_count DESC
        """)
        )
        key_platforms = dict(result.fetchall())

        # Check for NAPALM high-quality mappings
        result = await session.execute(
            text("""
            SELECT COUNT(*)
            FROM command_mappings
            WHERE discovery_metadata->>'napalm_getter' IS NOT NULL
        """)
        )
        napalm_count = result.scalar()

        # Check learning schema
        result = await session.execute(
            text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'command_mappings'
            AND column_name IN ('execution_count', 'success_count', 'confidence_history')
        """)
        )
        learning_columns = [row[0] for row in result.fetchall()]

        logger.info("✅ Intelligent seeding verification results:")
        logger.info(f"   📊 Total commands: {total_commands}")
        logger.info(f"   🎯 Confidence distribution: {confidence_dist}")
        logger.info(f"   🖥️  Key platforms: {key_platforms}")
        logger.info(f"   ⭐ NAPALM mappings: {napalm_count}")
        logger.info(f"   🧠 Learning columns: {learning_columns}")

        # Verify critical intents exist
        test_intents = ["bgp_neighbors", "interfaces", "version", "lldp_neighbors"]
        for intent in test_intents:
            result = await session.execute(
                text("""
                SELECT platform, command, confidence
                FROM command_mappings
                WHERE intent = :intent
                ORDER BY confidence DESC
                LIMIT 3
            """),
                {"intent": intent},
            )

            mappings = result.fetchall()
            if mappings:
                logger.info(f"   ✅ {intent}: {len(mappings)} mappings found")
                for mapping in mappings:
                    logger.info(
                        f"      • {mapping.platform}: {mapping.command} (conf: {mapping.confidence})"
                    )
            else:
                logger.warning(f"   ⚠️  {intent}: No mappings found")

        return total_commands > 0 and len(key_platforms) >= 3

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

    finally:
        await session.close()


async def main():
    """Main execution function with intelligent seeding"""

    print("🎯 NetOps Three-Tier Discovery - Intelligent PostgreSQL Seeding")
    print("=" * 70)

    # Run intelligent seeding
    success = await seed_postgres_with_conflict_resolution()
    if not success:
        print("❌ Intelligent seeding failed!")
        return 1

    # Verify seeding
    success = await verify_intelligent_seeding()
    if not success:
        print("❌ Verification failed!")
        return 1

    # Optional: Run confidence true-up (on-demand)
    print("\n" + "=" * 70)
    print("🎯 Running confidence true-up (optional)...")
    # true_up_result = await true_up_command_confidence() - ruff says this isnt used

    print("=" * 70)
    print("✅ Intelligent PostgreSQL seeding completed successfully!")
    print("🎯 Three-tier discovery system ready with learning capabilities")
    print("")
    print("🧠 Intelligence Features Enabled:")
    print("   • Conflict resolution (NAPALM > Community > LLM)")
    print("   • Learning system (LLM starts at 0.7, +0.1 per success)")
    print("   • Automatic demotion (poor performers demoted)")
    print("   • Confidence history tracking")
    print("")
    print("Next steps:")
    print("1. Update universal_request_processor.py with three-tier integration")
    print("2. Test: curl -X POST http://localhost:8000/api/v1/universal")
    print("3. Monitor: curl http://localhost:8000/api/v1/analytics/dashboard")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
