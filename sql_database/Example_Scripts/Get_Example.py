from sql_api import SqlConnect
import os


#######################################
#### TO BE USED AS AN EXAMPLE ONLY ####
#######################################

sql = SqlConnect(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer", username="ctrl-alt-elite",
                 password="Tsnte7TF6nMZTPY")

sql.connect()

keys = {"density":1}

print(sql.get_data("dbo.ddsm_dataset", top_rows="20", displayed_columns=('id','density'), keys=keys))


sql.close()
