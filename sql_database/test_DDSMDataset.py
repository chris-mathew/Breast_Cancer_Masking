from database_connect import DDSMDataset

def test_1_DDSMDataset():
    data = DDSMDataset()
    #1.Testing single data
    assert data.get_grouped_data(0,single=True)['group_id'] == 1
    
def test_2_DDSMDataset():
    data = DDSMDataset()
    #2.Testing group data
    assert data.get_grouped_data(3)[0]['group_id'] == 4
    
def test_3_DDSMDataset():
    data = DDSMDataset()
    #3.Testing get_length function
    assert data.get_length() == 106
    