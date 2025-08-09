import traceback
import sys


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_details
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_details: sys):
        """
        Returns a detailed error message with traceback information.
        """
        _, _, tb = error_details.sys.exc_info()
        file_name = tb.tb_frame.f_code.co_filename
        line_number = tb.tb_lineno

        return f'Error in {file_name} at line {line_number}: {error_message}'

    def __str__(self):
        return self.error_message