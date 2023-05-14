import unittest
import cv2
from app_model import test_function_for_bb_detections
import logging
import warnings

logging.basicConfig(filename='detecton_logs.log', format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_image(i):
    path = "./Verification/"+str(i)+".jpeg"
    return cv2.imread(path)

def get_image_jpg(i):
    path = "./Verification/"+str(i)+".jpg"
    return cv2.imread(path)

print("Beginning Testing ...\n")

class TestModel(unittest.TestCase):
    def test1(self):
        text_list, bb_confidences = test_function_for_bb_detections(get_image(2))
        self.assertEqual(text_list[0], "MH20BY3665")
        self.assertNotEqual(text_list[0], -1)
        if text_list[0]==-1:
            print("No Number Plate found in Image-1")
            logger.info("No Number Plate found in Image-1")
        else:
            print("Bounding box of Image-1 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
            logger.info("Bounding box of Image-1 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])          

    def test2(self):
        text_list, bb_confidences = test_function_for_bb_detections(get_image(4))
        self.assertEqual(text_list[0], "MH14EU3498")
        self.assertNotEqual(text_list[0], -1)
        if text_list[0]==-1:
            print("No Number Plate found in Image-2")
            logger.info("No Number Plate found in Image-2")
        else:
            print("Bounding box of Image-2 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
            logger.info("Bounding box of Image-2 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])

#     def test3(self):
#         text_list, bb_confidences = test_function_for_bb_detections(get_image(9))
#         self.assertEqual(text_list[0], "HR26U7501")
#         self.assertNotEqual(text_list[0], -1)
#         if text_list[0]==-1:
#             print("No Number Plate found in Image-3")
#             logger.info("No Number Plate found in Image-3")
#         else:
#             print("Bounding box of Image-3 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
#             logger.info("Bounding box of Image-3 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])

    def test4(self):
        text_list, bb_confidences = test_function_for_bb_detections(get_image(6))
        self.assertEqual(text_list, -1)
        self.assertNotEqual(text_list, "SaiKsk")
        if text_list==-1:
            print("No Number Plate found in Image-4")
            logger.info("No Number Plate found in Image-4")   
        else:
            print("Bounding box of Image-4 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
            logger.info("Bounding box of Image-4 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])

    def test5(self):
        text_list, bb_confidences = test_function_for_bb_detections(get_image(8))
        self.assertEqual(text_list[0], "DL3CAY9324")
        self.assertNotEqual(text_list, -1)
        if text_list==-1:
            print("No Number Plate found in Image-5")
            logger.info("No Number Plate found in Image-5")
        else:
            print("Bounding box of Image-5 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
            logger.info("Bounding box of Image-5 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])

#     def test6(self):
#         text_list, bb_confidences = test_function_for_bb_detections(get_image(7))
#         self.assertEqual(text_list[0], "GJW115A1138")
#         self.assertNotEqual(text_list, -1)
#         if text_list==-1:
#             print("No Number Plate found in Image-6")
#             logger.info("No Number Plate found in Image-6")
#         else:
#             print("Bounding box of Image-6 has been predicted with an accuracy of", bb_confidences[0], "and the detected Number plate text is", text_list[0])
#             logger.info("Bounding box of Image-6 has been predicted with an accuracy of %f and the detected Number plate text is %s",bb_confidences[0],  text_list[0])

if __name__ == '__main__':
    unittest.main()
