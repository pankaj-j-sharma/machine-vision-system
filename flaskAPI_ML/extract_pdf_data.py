"""
Auther: Shailesh S Sarda
"""

#To extract the pdf text and store in the tabular form [Smart Document Reader]
## 1. Code for Shop Finding Report Document
"""

!pip install pdf2image
!apt-get install poppler-utils
!pip install easyocr
"""
from pdf2image import convert_from_path
import easyocr
import numpy as np
import PIL
from PIL import ImageDraw
import pandas as pd
import re

# reader = easyocr.Reader(['en'])

# pdf_path = "/content/drive/MyDrive/SIAEC/20490973RMI_SFR.pdf"

# images = convert_from_path(pdf_path)

# from IPython.display import display, Image
# display(images[0])

# bounds = reader.readtext(np.array(images[0]), 
#                                   min_size = 0,
#                                   slope_ths = 0.2,
#                                   ycenter_ths = 0.7,
#                                   height_ths = 0.6,
#                                   width_ths = 0.8,
#                                   decoder = "beamsearch",
#                                   beamWidth = 10)

# def draw_boxes(image, bounds, color="red", width=2):
#   draw = ImageDraw.Draw(image)
#   for bound in bounds:
#     p0, p1, p2, p3 = bound[0]
#     draw.line([*p0, *p1, *p2, *p3], fill= color, width =width )
#   return image

# draw_boxes(images[0], bounds)

# text = ''


# for i in range(len(bounds)):
#   text = text + bounds[i][1] + '\n'
  
# # print(text)

# company_name = bounds[0][1]
# company_name

# customer = bounds[11][1]
# customer

# import re

# start_1 = text.find("Reason for removal (reported by customer):") + len("Reason for removal (reported by customer):")
# end_1 = text.find("Incoming inspection")
# substring_1 = text[start_1:end_1]
# # print(substring_1)

# start_2 = text.find("Incoming inspection") + len("Incoming inspection")
# end_2 = text.find("Shop findings")
# substring_2 = text[start_2:end_2]
# # print(substring_2)


# start_3 = text.find("Shop findings") + len("Shop findings")
# end_3 = text.find("Works to be performed")
# substring_3 = text[start_3:end_3]
# # print(substring_3)


# start_4 = text.find("Works to be performed") + len("Works to be performed")
# end_4 = text.find("Shop manager name")
# substring_4 = text[start_4:end_4]
# # print(substring_4)


# start_5 = text.find("Shop manager name") + len("Shop manager name")
# end_5 = text.find("Page 1/1")
# substring_5 = text[start_5:end_5]
# # print(substring_5)


# start_customer_repair_order = text.find("Customer Repair Order") + len("Customer Repair Order")
# end_customer_repair_order = text.find("Incoming part number")
# substring_customer_repair_order = text[start_customer_repair_order:end_customer_repair_order]
# # print(substring_customer_repair_order)

# start_Incoming_part_number = text.find("Incoming part number") + len("Incoming part number")
# end_Incoming_part_number = text.find("Serial Number")
# substring_Incoming_part_number = text[start_Incoming_part_number:end_Incoming_part_number]
# # print(substring_Incoming_part_number)

# start_Serial_Number = text.find("Serial Number") + len("Serial Number")
# end_Serial_Number = text.find("Description_")
# substring_Serial_Number = text[start_Serial_Number:end_Serial_Number]
# # print(substring_Serial_Number)


# start_Description_ = text.find("Description_") + len("Description_")
# end_Description_ = text.find("Aircraft registration_")
# substring_Description_ = text[start_Description_ : end_Description_]
# # print(substring_Description_)

pdf_stack = ["/content/drive/MyDrive/SIAEC/20490973RMI_SFR.pdf", 
             "/content/drive/MyDrive/SIAEC/20481279RTTW_SFR.pdf"]


Company_Name = []
Customer = []
Reason_For_Removal = []
Customer_Repair_Order = []
Incoming_part_number = []
Serial_Number = []
Description_ = []
Incoming_Inspection = []
Shop_Findings = []
Work_To_Be_Performed = []
Shop_manager_name = []


for pdfs in pdf_stack:
  reader = easyocr.Reader(['en'])
  images = convert_from_path(pdfs)
  bounds = reader.readtext(np.array(images[0]), 
                                  min_size = 0,
                                  slope_ths = 0.2,
                                  ycenter_ths = 0.7,
                                  height_ths = 0.6,
                                  width_ths = 0.8,
                                  decoder = "beamsearch",
                                  beamWidth = 10)
  text = ''
  for i in range(len(bounds)):
    text = text + bounds[i][1]


  company_name = bounds[0][1]
  Company_Name.append(company_name)

  customer = bounds[11][1]
  Customer.append(customer)

  start_customer_repair_order = text.find("Customer Repair Order") + len("Customer Repair Order")
  end_customer_repair_order = text.find("Incoming part number")
  substring_customer_repair_order = text[start_customer_repair_order:end_customer_repair_order]
  Customer_Repair_Order.append(substring_customer_repair_order)


  start_Incoming_part_number = text.find("Incoming part number") + len("Incoming part number")
  end_Incoming_part_number = text.find("Serial Number")
  substring_Incoming_part_number = text[start_Incoming_part_number:end_Incoming_part_number]
  Incoming_part_number.append(substring_Incoming_part_number)

  start_Serial_Number = text.find("Serial Number") + len("Serial Number")
  end_Serial_Number = text.find("Description_")
  substring_Serial_Number = text[start_Serial_Number:end_Serial_Number]
  Serial_Number.append(substring_Serial_Number)

  start_Description_ = text.find("Description_") + len("Description_")
  end_Description_ = text.find("Aircraft registration_")
  substring_Description_ = text[start_Description_ : end_Description_]
  Description_.append(substring_Description_)

  start_1 = text.find("Reason for removal (reported by customer):") + len("Reason for removal (reported by customer):")
  end_1 = text.find("Incoming inspection")
  substring_1 = text[start_1:end_1]
  Reason_For_Removal.append(substring_1)

  start_2 = text.find("Incoming inspection") + len("Incoming inspection")
  end_2 = text.find("Shop findings")
  substring_2 = text[start_2:end_2]
  Incoming_Inspection.append(substring_2)


  start_3 = text.find("Shop findings") + len("Shop findings")
  end_3 = text.find("Works to be performed")
  substring_3 = text[start_3:end_3]
  Shop_Findings.append(substring_3)


  start_4 = text.find("Works to be performed") + len("Works to be performed")
  end_4 = text.find("Shop manager name")
  substring_4 = text[start_4:end_4]
  Work_To_Be_Performed.append(substring_4)

  start_5 = text.find("Shop manager name") + len("Shop manager name")
  end_5 = text.find("Page 1/1")
  substring_5 = text[start_5:end_5]
  Shop_manager_name.append(substring_5)


# initialise data of lists.
data = {'Company Name':Company_Name,
        'Customer':Customer,
        'Reason for Removal': Reason_For_Removal,
        'Customer Repair Order': Customer_Repair_Order,
        'Incoming Part Number': Incoming_part_number,
        'Serial Number': Serial_Number,
        'Description': Description_,
        'Incoming Inspection':Incoming_Inspection,
        'Shop Findings':Shop_Findings,
        'Work To be Performed':Work_To_Be_Performed,
        'Shop Manager Name': Shop_manager_name}
 
# Create DataFrame
df = pd.DataFrame(data)
 
# Print the output.
df.to_excel("/content/drive/MyDrive/SIAEC/sample.xlsx")