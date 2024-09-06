
from AutoMobilePriceRegression.config.configuration import ConfigurationManager
from AutoMobilePriceRegression.components.data_transformation import DataTransfornmation
from AutoMobilePriceRegression.utils.common import logger


STAGE_NAME = "data transformation stage"

class DataTransfornmationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransfornmation(data_transformation_config)
        data_transformation.train_test_splitting()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransfornmationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        