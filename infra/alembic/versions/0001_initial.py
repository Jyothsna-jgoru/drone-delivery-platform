"""Initial durable operational schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

TABLES = ["simulation_runs","simulation_episodes","decision_events","drone_messages","collision_risk_events","safety_interventions","telemetry_samples","agent_executions","tool_calls","approval_requests","training_runs","model_versions","evaluation_results","incidents","incident_timeline_events"]

def upgrade():
    op.create_table("drones", sa.Column("id",sa.String(36),primary_key=True),sa.Column("created_at",sa.DateTime(timezone=True)),sa.Column("name",sa.String(64),unique=True),sa.Column("status",sa.String(32)))
    op.create_table("packages",sa.Column("id",sa.String(36),primary_key=True),sa.Column("created_at",sa.DateTime(timezone=True)),sa.Column("weight",sa.Float()),sa.Column("priority",sa.String(16)),sa.Column("destination",sa.JSON()))
    op.create_table("missions",sa.Column("id",sa.String(36),primary_key=True),sa.Column("created_at",sa.DateTime(timezone=True)),sa.Column("status",sa.String(32)),sa.Column("request",sa.JSON()))
    op.create_table("mission_assignments",sa.Column("id",sa.String(36),primary_key=True),sa.Column("created_at",sa.DateTime(timezone=True)),sa.Column("mission_id",sa.String(36)),sa.Column("drone_id",sa.String(36)),sa.Column("package_id",sa.String(36)))
    for table in TABLES:
        op.create_table(table,sa.Column("id",sa.String(36),primary_key=True),sa.Column("created_at",sa.DateTime(timezone=True)),sa.Column("payload",sa.JSON(),nullable=False))

def downgrade():
    for table in reversed(TABLES): op.drop_table(table)
    op.drop_table("mission_assignments")
    op.drop_table("missions")
    op.drop_table("packages")
    op.drop_table("drones")

