# convert pascal voc to yolov5
from pylabel import importer
import os


class ConvertAnnotation:
    def __init__(self, srcAnnotation='PascalVOC', dstAnnotation='Yolov5', base_path=os.getcwd(), name='tmpConversion', src_annotation_path="labels", src_images_path="images", dst_annotation_path="converted"):
        self.srcAnnotation = srcAnnotation
        self.dstAnnotation = dstAnnotation
        self.base_path = base_path
        self.name = name
        self.path_to_annotations = os.path.join(base_path, src_annotation_path)
        self.path_to_images = os.path.join(base_path, src_images_path)
        self.path_to_annotations_export = os.path.join(
            base_path, src_annotation_path, dst_annotation_path)

    def convert(self):
        self.dataset = importer.ImportVOC(
            path=self.path_to_annotations, path_to_images=self.path_to_images, name=self.name)
        self.dataset.export.ExportToYoloV5(
            output_path=self.path_to_annotations_export)


if __name__ == '__main__':
    objAnnotConvert = ConvertAnnotation()
