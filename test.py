from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(num1,num2):
    try:
        result = num1 / num2
        logger.info(f"Dividing {num1} by {num2} gives {result}")
        return result
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise CustomException("Custom Error zero", sys)
    
if __name__ == "__main__":
    try:
        logger.info("Starting division operation")
        divide_numbers(10, 0)
    except CustomException as ce:
        logger.error(str(ce))