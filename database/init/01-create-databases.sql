-- ~/net-chatbot/database/init/01-create-databases.sql
-- Initialize databases for NetOps Three-Tier Discovery System

-- Create NetOps application database (already created by POSTGRES_DB)
-- But ensure it exists
SELECT 'CREATE DATABASE netops_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'netops_db')\gexec

-- Grant permissions to netops user
GRANT ALL PRIVILEGES ON DATABASE netops_db TO netops;

-- Connect to netops_db for further setup
\c netops_db netops

-- Create extensions for better performance
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set up timezone
SET timezone = 'UTC';

COMMENT ON DATABASE netops_db IS 'NetOps ChatBot Three-Tier Discovery System Database';
