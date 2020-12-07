# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Store testcases for global use

import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def testcase1():
    passage = "病患於109年10月5日入院急診。"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase2():
    passage = "病患於民國108年10月5日至本院入院急診，經手術之後，民國108年10月7日出院。"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase3():
    passage = "患者於民國109年01月20日08時20分急診就醫，經縫合手術治療後於民國109年01月20日10時50分出院。"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase4():
    passage = "病患於民國108年10月5日住院接受既定化學(Lipodox,Endoxan)治療，並於2020年05月05日出院,共住院02日。患者於2020/04/13,2020/05/04,共門診02次。"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase5():
    passage = "病患因上述原因曾於108年12月17日,108年12月26日,109年01月14日,109年02月04日,109年02月26日,109年03月24日,109年04月14日,109年05月05日,109年05月19日,109年05月28日至本院門診治療。曾於109年01月17日至109年01月23日,109年02月11日至109年02月14日,109年03月07日至109年03月10日,109年03月28日至109年03月31日,109年04月18日至109年04月21日,109年05月09日至109年05月12日住院並接受靜脈注射化學治療。(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase6():
    passage = "病患因上述原因,於民國109年05月18日至本院腫瘤醫學部一般病房住院,因病情需要於民國109年05月18日接受靜脈注射全身性免疫藥物與標靶藥物治療,於民國109年05月20日出院,宜於門診持續追蹤治療--以下空白--"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase7():
    passage = "病患曾於109年04月19日12:22~109年04月19日16:00至本院急診治療,於109年4月19日入院,109年4月22日行冠狀動脈繞道手術,109年4月22日至109年4月25日於加護病房治療,109年5月7日出院.出院後宜門診追蹤治療.(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase8():
    passage = "患者於109年05月20日至本院門診檢查接受治療至109年05月30日止,共計門診11次。(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase9():
    passage = "108.3.21108.4.8108.6.10108.9.9108.10.7108.12.2109.2.24109.5.18109.6.1門診"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase10():
    passage = "病患因上述病因於109年05月13日行右腕正中神經減壓手術,於109年05月14日,於109年06月01日神經外科門診追蹤治療--以下空白--"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase11():
    passage = "1,患者因前述原因,於2020-05-25至2020-05-26,共住院2日,住院接受注射標靶藥物治療。(以下空白)2,依病歷記錄,患者接受陳達人醫師於2020-05-25之本院門診追蹤治療,共計1次。(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase12():
    passage = "患者於109年5月27日來本院急診,自109年5月27日起至109年6月4日止來本院住院共9天,需繼續門診治療及療養。[以下空白]"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase13():
    passage = "患者於109年5月19日來本院急診。18時31分到本院急診就醫。給予腹部及胸部電腦斷層掃描,X光檢查,處置傷口及執行傷口縫合術(共1針)。宜休息3日及外科門診追蹤治療。[以下空白]"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase14():
    passage = "患者於民國109年04月04日經急診住院加護病房觀察治療後,109年04月06日病情改善,家屬要求自動出院自動出院.(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase15():
    passage = "住院日自108年09月09日至108年09月14日。[以下空白]"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase16():
    passage = "患者在108年12月25日07時42分至本院急診治療，經治療後，在108年12月25日住院，至108年12月27日出院。患者在108年12月30日至本院門診就醫治療。(以下空白)"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase17():
    passage = "病患因上述原因，於2020年4月26日住院，4月27日接受人工血管置放手術，4月28日接受靜脈注射全身性化學治療，4月29日出院，宜於門診持續追蹤治療--以下空白--"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase18():
    passage = "病患於2019年5月18日住院接受痔瘡手術。"
    return {"passage": passage}


@pytest.fixture(scope="session")
def testcase19():
    passage = "病患於1090810,1090811急診就診,1090811入院並接受骨折復位鋼針固定手術,1090813出院,共計住院3天,1090819門診追蹤治療。預計骨癒合約需三個月,專人看護一個月,需休養及吊帶使用並避免手臂肩膀活動三個月,證明用。(以下空白)"
    return {"passage": passage}
