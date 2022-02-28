# from steel_defect_detection import predict_steel_defects
# from surface_defects_detection import predict_surface_defects
# from metal_casting_defects import predict_metal_cast_defects
# from hard_hat_detection import predict_hard_hat_present
# from package_inspection import predict_packaging_defects


from prediction_setup import inferenceWithUserModel


def predictfrommodel(modelName, fileList):
    if modelName == 'Surface Defects':
        from surface_defects_detection import predict_surface_defects
        return predict_surface_defects(fileList)
    elif modelName == 'Metal Casting Defects':
        from metal_casting_defects import predict_metal_cast_defects
        return predict_metal_cast_defects(fileList)
    elif modelName == 'Hard Hat Present':
        from hard_hat_detection import predict_hard_hat_present
        return predict_hard_hat_present(fileList)
    elif modelName == 'Steel Defects':
        from steel_defect_detection import predict_steel_defects
        return predict_steel_defects(fileList)
    elif modelName == 'Package Damage Detection':
        from package_inspection import predict_packaging_defects
        return predict_packaging_defects(fileList)
