"""
test iotools for panond
"""

from pvlib.iotools import read_panond, parse_panond
from pvlib.tests.conftest import DATA_DIR
import io
# Not sure if I am creating these test scenarios correctly

fn_file = DATA_DIR / 'CPS SCH275KTL-DO-US-800-250kW_275kVA_1.OND'
ond_file = read_panond(fn_file)

fn_str = """PVObject_=pvGInverter
  Comment=ChintPower	CPS SCH275KTL-DO/US-800	Manufacturer 2020
  Version=6.81
  ParObj1=2020
  Flags=$00381562

  PVObject_Commercial=pvCommercial
    Comment=www.chintpower.com  (China)
    Flags=$0041
    Manufacturer=ChintPower
    Model=CPS SCH275KTL-DO/US-800
    DataSource=Manufacturer 2020
    YearBeg=2020
    Width=0.680
    Height=0.337
    Depth=1.100
    Weight=95.000
    NPieces=0
    PriceDate=02/06/20 00:02
    Currency=EUR
    Remarks, Count=2
      Str_1=Protection: -30 - +60, IP 66:  outdoor installable
      Str_2
    End of Remarks
  End of PVObject pvCommercial
  Transfo=Without

  Converter=TConverter
    PNomConv=250.000
    PMaxOUT=250.000
    VOutConv=800.0
    VMppMin=500
    VMPPMax=1500
    VAbsMax=1500
    PSeuil=500.0
    EfficMax=99.01
    EfficEuro=98.49
    FResNorm=0.00
    ModeOper=MPPT
    CompPMax=Lim
    CompVMax=Lim
    MonoTri=Tri
    ModeAffEnum=Efficf_POut
    UnitAffEnum=kW
    PNomDC=253.000
    PMaxDC=375.000
    IDCMax=0.0
    IMaxDC=360.0
    INomAC=181.0
    IMaxAC=199.0
    TPNom=45.0
    TPMax=40.0
    TPLim1=50.0
    TPLimAbs=60.0
    PLim1=225.000
    PLimAbs=90.000
    PInEffMax =150000.000
    PThreshEff=3332.4
    HasdefaultPThresh=False

    ProfilPIO=TCubicProfile
      NPtsMax=11
      NPtsEff=9
      LastCompile=$8085
      Mode=1
      Point_1=1250,0
      Point_2=7500,6923
      Point_3=12500,11875
      Point_4=25000,24250
      Point_5=50000,49100
      Point_6=75000,73875
      Point_7=150000,148515
      Point_8=250000,246500
      Point_9=275000,270325
      Point_10=0,0
      Point_11=0,0
    End of TCubicProfile
    VNomEff=880.0,1174.0,1300.0,
    EfficMaxV=98.260,99.040,98.860,
    EfficEuroV=97.986,98.860,98.661,

    ProfilPIOV1=TCubicProfile
      NPtsMax=11
      NPtsEff=9
      LastCompile=$8089
      Mode=1
      Point_1=300.0,0.0
      Point_2=13012.7,12500.0
      Point_3=25720.2,25000.0
      Point_4=51093.4,50000.0
      Point_5=76437.0,75000.0
      Point_6=127213.5,125000.0
      Point_7=190995.2,187500.0
      Point_8=255440.9,250000.0
      Point_9=281301.1,275000.0
      Point_10=0.0,0.0
      Point_11=0.0,0.0
    End of TCubicProfile

    ProfilPIOV2=TCubicProfile
      NPtsMax=11
      NPtsEff=9
      LastCompile=$8089
      Mode=1
      Point_1=300.0,0.0
      Point_2=12850.8,12500.0
      Point_3=25401.3,25000.0
      Point_4=50581.7,50000.0
      Point_5=75795.9,75000.0
      Point_6=126211.6,125000.0
      Point_7=189623.8,187500.0
      Point_8=253138.9,250000.0
      Point_9=278763.3,275000.0
      Point_10=0.0,0.0
      Point_11=0.0,0.0
    End of TCubicProfile

    ProfilPIOV3=TCubicProfile
      NPtsMax=11
      NPtsEff=9
      LastCompile=$8089
      Mode=1
      Point_1=300.0,0.0
      Point_2=12953.4,12500.0
      Point_3=25512.8,25000.0
      Point_4=50679.1,50000.0
      Point_5=75895.6,75000.0
      Point_6=126441.4,125000.0
      Point_7=189835.0,187500.0
      Point_8=253472.6,250000.0
      Point_9=279017.9,275000.0
      Point_10=0.0,0.0
      Point_11=0.0,0.0
    End of TCubicProfile
  End of TConverter
  NbInputs=36
  NbMPPT=12
  TanPhiMin=-0.750
  TanPhiMax=0.750
  NbMSInterne=2
  MasterSlave=No_M_S
  IsolSurvey =Yes
  DC_Switch=Yes
  MS_Thresh=0.8
  Night_Loss=5.00
End of PVObject pvGcomperter
"""
f_obj = io.StringIO(fn_str)
ond_str = parse_panond(f_obj)

fn_file = DATA_DIR / 'ET-M772BH550GL.PAN'
mod_file = read_panond(fn_file)

fn_str = """PVObject_=pvModule
  Version=7.2
  Flags=$00900243

  PVObject_Commercial=pvCommercial
    Comment=ET SOLAR
    Flags=$0041
    Manufacturer=ET SOLAR
    Model=ET-M772BH550GL
    DataSource=Manufacturer 2021
    YearBeg=2021
    Width=1.134
    Height=2.278
    Depth=0.035
    Weight=32.000
    NPieces=100
    PriceDate=06/04/22 12:39
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=72
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsARCoating
  GRef=1000
  TRef=25.0
  PNom=550.0
  PNomTolUp=0.90
  BifacialityFactor=0.700
  Isc=14.000
  Voc=49.90
  Imp=13.110
  Vmp=41.96
  muISC=7.28
  muVocSpec=-128.0
  muPmpReq=-0.340
  RShunt=300
  Rp_0=2000
  Rp_Exp=5.50
  RSerie=0.203
  Gamma=0.980
  muGamma=-0.0001
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=16.716
  RDiode=0.010
  VRevDiode=-0.70
  IMaxDiode=30.0
  AirMassRef=1.500
  CellArea=165.1
  SandiaAMCorr=50.000

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=20.0,1.00000
      Point_3=30.0,1.00000
      Point_4=40.0,0.99000
      Point_5=50.0,0.98000
      Point_6=60.0,0.96000
      Point_7=70.0,0.89000
      Point_8=80.0,0.66000
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM
End of PVObject pvModule
"""
f_obj = io.StringIO(fn_str)
mod_str = parse_panond(f_obj)
