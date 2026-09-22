from src.mlops.logging.logger import get_logger


logger = get_logger("MLOpsTest")


logger.info("FraudSentinel MLOps logging test started.")
logger.info("Logging system is operational.")
logger.warning("This is a test warning.")
logger.info("FraudSentinel MLOps logging test completed.")