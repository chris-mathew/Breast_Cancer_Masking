DECLARE @Table NVARCHAR(255) = 'ddsm_dataset'
DECLARE @Run_String NVARCHAR(MAX)
SET @Run_String = 'TRUNCATE TABLE '+ QUOTENAME(@Table)
EXEC sp_executesql @Run_String