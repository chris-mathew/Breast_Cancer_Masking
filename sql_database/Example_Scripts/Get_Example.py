import sql_api
import os


#######################################
#### TO BE USED AS AN EXAMPLE ONLY ####
#######################################

sql = sql_api.SqlConnect(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer", username="ctrl-alt-elite",
                 password="Tsnte7TF6nMZTPY")

#Connect to the SQL server
sql.connect()
#Return images that has a density of 1
keys = {"density":1}
#Prints the rows that have a density of 1
print(sql.get_data("dbo.ddsm_dataset", top_rows="20", displayed_columns=('id','density'), keys=keys))

#Close the connection
sql.close()
