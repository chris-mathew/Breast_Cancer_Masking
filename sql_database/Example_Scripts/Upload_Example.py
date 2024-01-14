from sql_api import SqlConnect
import os


#######################################
#### TO BE USED AS AN EXAMPLE ONLY ####
#######################################

sql = SqlConnect(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer", username="ctrl-alt-elite",
                 password="Tsnte7TF6nMZTPY")

sql.connect()
path = "C:/Users/chris/OneDrive - Imperial College London/CBIS Dataset/manifest-Egq0PU078220738724328010106/CBIS-DDSM/Grouped" #path to the data which is a set of folders containing the 4 different views

folder_names = os.listdir(path) #
values = []
group_id = 0

for dir in folder_names:
    group_id+=1
    dirnames = os.listdir(path+"/"+dir)
    for folder in dirnames:
        folder_value = {}
        folder_value['group_id'] = group_id
        with open(path+"/"+dir+'/'+folder, 'rb') as file:
            folder_value['pixel_data'] = file.read()
        image_name_split = folder.split(".")[0].split("_")
        folder_value['patient_id'] = int(image_name_split[1])
        if image_name_split[2] == 'LEFT':
            folder_value['direction'] = 0
        else:
            folder_value['direction'] = 1
        if image_name_split[3] == "CC":
            folder_value['image_view'] = 0
        else:
            folder_value['image_view'] = 1
        folder_value['density'] = int(image_name_split[4])

        sql.insert("dbo.ddsm_dataset", [folder_value])


sql.close()