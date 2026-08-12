"""
Octavian Ticker Universe — Single Source of Truth
All tickers, sectors, aliases, and universe helpers in one place.
Every other module should import from here instead of maintaining its own lists.
"""

import json
import os
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

_CACHE_PATH = Path(__file__).parent / "ticker_universe_cache.json"
_LOCK = threading.Lock()
_universe_instance: Optional["TickerUniverse"] = None


#
# SEED TICKERS — ~2000 hardcoded across every sector
#

_SEED_BY_SECTOR: Dict[str, List[str]] = {
    #  TECHNOLOGY
    "technology": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "META",
        "NVDA",
        "AVGO",
        "ADBE",
        "CRM",
        "ORCL",
        "CSCO",
        "ACN",
        "IBM",
        "INTC",
        "AMD",
        "QCOM",
        "TXN",
        "NOW",
        "INTU",
        "AMAT",
        "ADI",
        "LRCX",
        "KLAC",
        "SNPS",
        "CDNS",
        "MRVL",
        "NXPI",
        "MCHP",
        "ON",
        "SWKS",
        "FTNT",
        "PANW",
        "CRWD",
        "ZS",
        "OKTA",
        "CYBR",
        "TENB",
        "RPD",
        "VRNS",
        "ANET",
        "TEAM",
        "WDAY",
        "HUBS",
        "DDOG",
        "SNOW",
        "MDB",
        "NET",
        "CFLT",
        "ESTC",
        "PLTR",
        "AI",
        "PATH",
        "GTLB",
        "S",
        "DOCN",
        "DT",
        "NEWR",
        "MU",
        "WOLF",
        "MPWR",
        "ALGM",
        "DIOD",
        "SLAB",
        "SITM",
        "RMBS",
        "FORM",
        "CRUS",
        "POWI",
        "SMTC",
        "ACLS",
        "OLED",
        "AMBA",
        "LSCC",
        "INDI",
        "MTSI",
        "GFS",
        "SMCI",
        "ARM",
        "IONQ",
        "RGTI",
        "QUBT",
        "ARQQ",
        "BBAI",
        "BIGC",
        "ASAN",
        "MNDY",
        "FROG",
        "BRZE",
        "CWAN",
        "JAMF",
        "PCTY",
        "PAYC",
        "BILL",
        "FOUR",
        "GLBE",
        "GLOB",
        "EPAM",
        "WIX",
        "SQSP",
        "WEAV",
        "VERX",
        "RELY",
        "RBRK",
        "SOUN",
        "LUNR",
        "ASTS",
        "RDW",
        "BKSY",
        "APLD",
        "MARA",
        "RIOT",
        "CLSK",
        "BTBT",
        "CIFR",
        "HUT",
        "BTDR",
        "CORZ",
        "WULF",
        "DELL",
        "HPQ",
        "HPE",
        "NTAP",
        "PSTG",
        "STX",
        "WDC",
        "CTSH",
        "LDOS",
        "SAIC",
        "BAH",
        "KD",
        "DXC",
        "PRFT",
        "IRBT",
        "BRKS",
        "CGNX",
        "TER",
        "NOVT",
        "MKSI",
        "COHR",
        "MANH",
        "APPF",
        "NCNO",
        "ZI",
        "CERT",
        "ALTR",
        "SUMO",
        "OPEN",
        "RDFN",
        "ZG",
        "Z",
        "TOST",
        "SQ",
        "SHOP",
    ],
    #  SEMICONDUCTORS (dedicated sub-sector)
    "semiconductors": [
        "NVDA",
        "AMD",
        "INTC",
        "AVGO",
        "QCOM",
        "TXN",
        "AMAT",
        "LRCX",
        "KLAC",
        "SNPS",
        "CDNS",
        "MRVL",
        "NXPI",
        "MCHP",
        "ON",
        "SWKS",
        "MU",
        "ADI",
        "WOLF",
        "MPWR",
        "ALGM",
        "GFS",
        "ARM",
        "SMCI",
        "TSM",
        "ASML",
        "UMC",
        "SLAB",
        "SITM",
        "RMBS",
        "DIOD",
        "POWI",
        "SMTC",
        "CRUS",
        "LSCC",
        "ACLS",
        "OLED",
        "AMBA",
        "INDI",
        "MTSI",
        "FORM",
        "COHR",
        "IPGP",
        "ENTG",
        "MKSI",
        "BRKS",
        "TER",
        "NOVT",
        "CGNX",
        "PLAB",
        "AOSL",
        "SYNA",
        "MACOM",
        "CEVA",
        "AMKR",
        "HIMX",
        "MXL",
        "SMTC",
    ],
    #  CYBERSECURITY
    "cybersecurity": [
        "PANW",
        "CRWD",
        "ZS",
        "OKTA",
        "CYBR",
        "FTNT",
        "TENB",
        "RPD",
        "VRNS",
        "S",
        "RBRK",
        "QLYS",
        "HACK",
        "CIBR",
        "BUG",
        "FEYE",
        "MNDT",
        "SAIL",
        "RDWR",
        "CHKP",
        "INSG",
        "SCWX",
        "OSPN",
        "EVBG",
        "TMEI",
        "PRGS",
        "SPSC",
        "FFIV",
        "JNPR",
        "NSIT",
        "CALX",
        "CWBR",
        "DTEX",
        "CLBT",
        "DLO",
        "NTCT",
        "SIFY",
        "MGNI",
        "AKAM",
        "LLNW",
        "FSLY",
        "NET",
        "GDDY",
        "TNET",
        "EGHT",
        "COMM",
        "CIEN",
        "INFN",
        "VIAV",
        "LITE",
        "IIVI",
        "AAOI",
    ],
    #  HEALTHCARE / PHARMA / BIOTECH
    "healthcare": [
        "LLY",
        "UNH",
        "JNJ",
        "MRK",
        "ABBV",
        "TMO",
        "ABT",
        "PFE",
        "AMGN",
        "GILD",
        "BMY",
        "MDT",
        "SYK",
        "BSX",
        "ISRG",
        "EW",
        "ZTS",
        "CI",
        "ELV",
        "HUM",
        "CNC",
        "MOH",
        "HCA",
        "UHS",
        "THC",
        "DVA",
        "GEHC",
        "BAX",
        "BDX",
        "A",
        "DHR",
        "IQV",
        "CRL",
        "MEDP",
        "WST",
        "TFX",
        "HOLX",
        "IDXX",
        "ALGN",
        "DXCM",
        "VRTX",
        "REGN",
        "MRNA",
        "BIIB",
        "ILMN",
        "ALNY",
        "SRPT",
        "BMRN",
        "RARE",
        "NBIX",
        "EXAS",
        "INCY",
        "JAZZ",
        "BNTX",
        "NVAX",
        "IONS",
        "PRTA",
        "ARWR",
        "FATE",
        "LEGN",
        "PCVX",
        "ACAD",
        "RVMD",
        "AXSM",
        "CRSP",
        "EDIT",
        "NTLA",
        "BEAM",
        "VERV",
        "DNA",
        "RXRX",
        "ABCL",
        "SANA",
        "RCKT",
        "KYMR",
        "DAWN",
        "DYNE",
        "NUVB",
        "TVTX",
        "KROS",
        "IMVT",
        "CRNX",
        "ERAS",
        "RLAY",
        "TGTX",
        "VRNA",
        "ACLX",
        "JANX",
        "VEEV",
        "DOCS",
        "TDOC",
        "HIMS",
        "GDRX",
        "OSCR",
        "ACCD",
        "SDGR",
        "PHR",
        "NTRA",
        "GH",
        "TWST",
        "OLINK",
        "CTKB",
        "PODD",
        "IRTC",
        "NVST",
        "GKOS",
        "NVCR",
        "INSP",
        "LIVN",
        "HAE",
        "MMSI",
        "ATEC",
        "TNDM",
        "MASI",
        "NUVA",
        "VTRS",
        "TEVA",
        "CTLT",
        "OGN",
        "ELAN",
        "SUPN",
        "PCRX",
        "AMPH",
        "CVS",
        "WBA",
        "CAH",
        "MCK",
        "COR",
    ],
    #  FINANCIALS
    "financials": [
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "USB",
        "PNC",
        "TFC",
        "SCHW",
        "BLK",
        "BX",
        "KKR",
        "APO",
        "ARES",
        "CG",
        "OWL",
        "AXP",
        "COF",
        "DFS",
        "SYF",
        "ALLY",
        "SPGI",
        "MCO",
        "ICE",
        "CME",
        "CBOE",
        "NDAQ",
        "FIS",
        "FISV",
        "GPN",
        "SQ",
        "PYPL",
        "V",
        "MA",
        "ADP",
        "PAYX",
        "WTW",
        "AON",
        "MMC",
        "AJG",
        "BRO",
        "RYAN",
        "MET",
        "PRU",
        "AIG",
        "ALL",
        "PGR",
        "TRV",
        "CB",
        "AFG",
        "WRB",
        "RNR",
        "ACGL",
        "HIG",
        "GL",
        "CINF",
        "ERIE",
        "COIN",
        "HOOD",
        "SOFI",
        "AFRM",
        "UPST",
        "UWMC",
        "RKT",
        "LPRO",
        "TOST",
        "LMND",
        "ROOT",
        "LC",
        "BILL",
        "FOUR",
        "FITB",
        "KEY",
        "RF",
        "HBAN",
        "MTB",
        "CFG",
        "ZION",
        "CMA",
        "FHN",
        "WAL",
        "EWBC",
        "FCNCA",
        "PNFP",
        "GBCI",
        "FNB",
        "BOKF",
        "SNV",
        "UMBF",
        "IBKR",
        "LPLA",
        "MKTX",
        "VIRT",
        "SBCF",
        "CADE",
        "ONB",
        "VLY",
        "COLB",
        "FFIN",
        "BPOP",
        "OFG",
        "KNSL",
        "PLMR",
        "PIPR",
        "EVR",
        "LAZ",
        "PJT",
        "HLI",
    ],
    #  CONSUMER DISCRETIONARY
    "consumer": [
        "AMZN",
        "TSLA",
        "HD",
        "LOW",
        "TJX",
        "ROST",
        "BURL",
        "TGT",
        "WMT",
        "COST",
        "DG",
        "DLTR",
        "FIVE",
        "OLLI",
        "BJ",
        "MCD",
        "SBUX",
        "CMG",
        "DPZ",
        "YUM",
        "QSR",
        "WING",
        "TXRH",
        "DRI",
        "EAT",
        "CAVA",
        "SHAK",
        "BROS",
        "NKE",
        "LULU",
        "DECK",
        "ONON",
        "BIRK",
        "SKX",
        "CROX",
        "VFC",
        "TPR",
        "CPRI",
        "RL",
        "PVH",
        "HBI",
        "LEVI",
        "ANF",
        "AEO",
        "GPS",
        "URBN",
        "ABNB",
        "BKNG",
        "EXPE",
        "MAR",
        "HLT",
        "H",
        "WYNN",
        "LVS",
        "MGM",
        "CZR",
        "DKNG",
        "PENN",
        "RSI",
        "GENI",
        "DIS",
        "CMCSA",
        "WBD",
        "PARA",
        "FOXA",
        "NFLX",
        "SPOT",
        "ROKU",
        "RBLX",
        "TTWO",
        "EA",
        "MTCH",
        "BMBL",
        "GM",
        "F",
        "RIVN",
        "LCID",
        "NIO",
        "LI",
        "XPEV",
        "GOEV",
        "VFS",
        "JOBY",
        "LILM",
        "ACHR",
        "BLDE",
        "STLA",
        "RACE",
        "BWA",
        "APTV",
        "LEA",
        "ALV",
        "GT",
        "ETSY",
        "W",
        "CHWY",
        "CVNA",
        "CARG",
        "DASH",
        "UBER",
        "LYFT",
        "GRAB",
        "PINS",
        "SNAP",
        "ZM",
        "DUOL",
        "COUR",
        "RH",
        "WSM",
        "VSCO",
        "SIG",
        "KSS",
        "JWN",
        "M",
        "DDS",
        "ULTA",
        "BBWI",
        "TPX",
    ],
    #  CONSUMER STAPLES
    "consumer_staples": [
        "PG",
        "KO",
        "PEP",
        "PM",
        "MO",
        "MDLZ",
        "CL",
        "KMB",
        "CHD",
        "CLX",
        "SJM",
        "GIS",
        "K",
        "CAG",
        "CPB",
        "HRL",
        "MKC",
        "HSY",
        "MNST",
        "KDP",
        "STZ",
        "BF-B",
        "SAM",
        "TAP",
        "DEO",
        "BUD",
        "CELH",
        "EL",
        "COTY",
        "IPAR",
        "ADM",
        "BG",
        "INGR",
        "DAR",
        "CALM",
        "FDP",
        "KR",
        "SFM",
        "GO",
        "ACI",
        "USFD",
        "PFGC",
        "SYY",
        "BTI",
        "TLRY",
        "CGC",
        "ACB",
        "CRON",
        "OGI",
        "SNDL",
        "TSN",
        "PPC",
        "JJSF",
        "POST",
        "LANC",
        "THS",
        "SMPL",
        "FRPT",
        "CENT",
        "CENTA",
        "SPB",
        "CWH",
        "WMK",
        "HAIN",
    ],
    #  ENERGY
    "energy": [
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "OXY",
        "MPC",
        "PSX",
        "VLO",
        "HES",
        "DVN",
        "FANG",
        "PXD",
        "HAL",
        "BKR",
        "MRO",
        "APA",
        "CTRA",
        "OVV",
        "PR",
        "AR",
        "RRC",
        "EQT",
        "SWN",
        "CHK",
        "MTDR",
        "CHRD",
        "SM",
        "NOG",
        "VTLE",
        "TRGP",
        "WMB",
        "OKE",
        "KMI",
        "ET",
        "EPD",
        "MPLX",
        "PAA",
        "AM",
        "DTM",
        "GPOR",
        "CIVI",
        "PTEN",
        "HP",
        "RIG",
        "DO",
        "VAL",
        "NE",
        "TDW",
        "DINO",
        "PBF",
        "CVI",
        "DK",
        "PARR",
        "FTI",
        "WHD",
        "LBRT",
        "OII",
    ],
    #  RENEWABLES / CLEAN ENERGY
    "clean_energy": [
        "ENPH",
        "FSLR",
        "SEDG",
        "RUN",
        "NOVA",
        "ARRY",
        "MAXN",
        "CSIQ",
        "JKS",
        "DQ",
        "PLUG",
        "BLDP",
        "BE",
        "CHPT",
        "EVGO",
        "BLNK",
        "STEM",
        "BEEM",
        "NEE",
        "AES",
        "CEG",
        "VST",
        "CWEN",
        "SHLS",
        "SPWR",
        "EOSE",
        "FLNC",
        "GNRC",
        "HASI",
        "CWEN-A",
        "ORA",
        "RNW",
        "AMPS",
        "VNET",
        "CLNE",
        "GEVO",
        "AMTX",
        "BEP",
        "BEPC",
        "CSAN",
        "REGI",
        "GPRE",
        "REX",
        "PTRA",
        "WKHS",
        "HYLN",
        "NKLA",
        "FCEL",
        "CLSK",
        "VELO",
        "OPAL",
        "AY",
        "TERP",
        "NOVA",
        "MAXN",
        "ENLT",
    ],
    #  NUCLEAR / URANIUM
    "uranium": [
        "CCJ",
        "UEC",
        "UUUU",
        "DNN",
        "NXE",
        "LEU",
        "SMR",
        "OKLO",
        "BWXT",
        "GEV",
        "URG",
        "UROY",
        "URA",
        "URNM",
        "NLR",
        "LTBR",
        "EU",
        "AEC",
        "FCU",
        "NMI",
        "GLO",
        "FIND",
        "AAZ",
        "ENCUF",
        "SRUUF",
        "PALAF",
        "FCUUF",
        "ANLDF",
        "BNNLF",
        "AZZUF",
        "GLATF",
        "DYLLF",
        "PENMF",
        "FNNZF",
        "BQSSF",
        "CVVUF",
        "MGAFF",
        "ELRFF",
        "LTSRF",
        "WSTRF",
        "PTU",
        "ISO",
        "SKYHF",
        "ISENF",
        "NCPCF",
        "FUU",
        "SYH",
        "AZZ",
        "VMY",
        "LOT",
    ],
    #  MINING
    "mining": [
        "NEM",
        "GOLD",
        "AEM",
        "FNV",
        "WPM",
        "RGLD",
        "KGC",
        "AGI",
        "PAAS",
        "HL",
        "EGO",
        "AU",
        "BTG",
        "IAG",
        "OR",
        "SSRM",
        "CDE",
        "NGD",
        "HMY",
        "DRD",
        "GFI",
        "SBSW",
        "SA",
        "MUX",
        "AG",
        "MAG",
        "SILV",
        "SVM",
        "FSM",
        "USAS",
        "EXK",
        "FCX",
        "SCCO",
        "TECK",
        "HBM",
        "ERO",
        "BHP",
        "RIO",
        "VALE",
        "NUE",
        "STLD",
        "CLF",
        "X",
        "AA",
        "CENX",
        "ATI",
        "RS",
        "CMC",
        "GDX",
        "GDXJ",
        "SIL",
        "COPX",
        "XME",
        "PICK",
        "REMX",
        "SAND",
        "GROY",
        "EMX",
        "ORVR",
        "ASA",
    ],
    #  MINERALS / RARE EARTH
    "minerals": [
        "MP",
        "UUUU",
        "ALB",
        "SQM",
        "LTHM",
        "PLL",
        "SGML",
        "LAC",
        "NTR",
        "MOS",
        "CF",
        "IPI",
        "ICL",
        "FMC",
        "CTVA",
        "VMC",
        "MLM",
        "CX",
        "EXP",
        "USLM",
        "TMRC",
        "URNM",
        "REMX",
        "HREE",
        "LYC",
        "ILUKA",
        "ARAFURA",
        "ASM",
        "PMET",
        "UCORE",
        "HBM",
        "ERO",
        "NEXA",
        "IVPAF",
        "CPER",
        "AMR",
        "BTU",
        "ARCH",
        "HCC",
        "ARLP",
        "TECK",
        "BHP",
        "RIO",
        "VALE",
        "SCCO",
        "FCX",
        "WPM",
        "FNV",
        "RGLD",
        "OR",
        "SAND",
        "EMX",
        "GROY",
        "ORVR",
        "MTA",
        "GXU",
    ],
    "rare_earth": [
        "MP",
        "UUUU",
        "REMX",
        "TMRC",
        "LYC",
        "LYSCF",
        "ASM",
        "ASMAF",
        "REEMF",
        "HREE",
        "UURAF",
        "MKR",
        "PMET",
        "UCORE",
        "UCU",
        "ILHMF",
        "ARAFF",
        "SHENGHE",
        "GXE",
        "VML",
        "VITOF",
        "NIOCF",
        "ARAF",
        "HAFNF",
        "QMC",
        "QMCI",
        "REE",
        "HAS",
        "NRSM",
        "MNEAF",
        "CYDVF",
        "PEMIF",
        "APPTF",
        "TOLRF",
        "AMYZF",
        "RSSFF",
        "MLLOF",
        "DEMRF",
        "HYRMF",
        "ABML",
        "CRECF",
        "ALLIF",
        "TLOFF",
        "MVDDF",
        "TSXVF",
        "FCSMF",
        "CXOXF",
        "SIRC",
        "SXOOF",
        "UAMY",
        "SMREF",
        "RARX",
        "RREE",
        "RVLWF",
    ],
    #  GOLD
    "gold": [
        "NEM",
        "GOLD",
        "AEM",
        "FNV",
        "WPM",
        "RGLD",
        "KGC",
        "AGI",
        "PAAS",
        "HL",
        "EGO",
        "AU",
        "BTG",
        "IAG",
        "OR",
        "SSRM",
        "CDE",
        "NGD",
        "HMY",
        "GFI",
        "GLD",
        "SGOL",
        "IAU",
        "DRD",
        "SA",
        "SBSW",
        "MUX",
        "SAND",
        "GROY",
        "EMX",
        "ORVR",
        "ASA",
        "BAR",
        "OUNZ",
        "PHYS",
        "OGC",
        "GLDG",
        "TORX",
        "DSVSF",
        "NFGFF",
        "SILG",
        "GDXJ",
        "GDX",
        "RING",
        "GOAU",
        "ELGXF",
        "NSRGF",
        "ECRTF",
        "WRLGF",
        "RGLSF",
    ],
    "silver": [
        "AG",
        "MAG",
        "SILV",
        "SVM",
        "FSM",
        "USAS",
        "EXK",
        "CDE",
        "PAAS",
        "HL",
        "WPM",
        "SLV",
        "PSLV",
        "SIVR",
        "SSRM",
        "MUX",
        "GPL",
        "SILJ",
        "SIL",
        "AUMN",
        "SLVRF",
        "ASM",
        "ABRA",
        "DSVSF",
        "RSNVF",
        "KNCRF",
        "FMCXF",
        "PRSDF",
        "MGMLF",
        "NUAG",
        "SLVP",
        "DSVS",
        "HAMRF",
        "ISVLF",
        "VZLA",
        "BCEKF",
        "ELOX",
        "BNCHF",
        "HYMC",
        "BKRRF",
        "SSVFF",
        "SRSIF",
        "PRSNF",
        "MMNGF",
    ],
    "copper": [
        "FCX",
        "SCCO",
        "TECK",
        "HBM",
        "ERO",
        "COPX",
        "BHP",
        "RIO",
        "VALE",
        "NEXA",
        "IVPAF",
        "CPER",
        "FM",
        "CS",
        "LUN",
        "OZL",
        "CMCLF",
        "ANFGF",
        "TCKRF",
        "IVNHF",
        "CPPMF",
        "QBCRF",
        "ZIJMF",
        "AAUKF",
        "KGHPF",
        "TGB",
        "TLRS",
        "NOVR",
        "ADZN",
        "FILO",
        "SOLG",
        "CUCO",
        "NGEX",
        "JOSM",
        "SWN",
        "COP",
        "WDOFF",
        "METCF",
        "AGLXY",
        "WLBMF",
        "CSCCF",
        "EROS",
        "HBMSF",
        "SMTS",
        "CAPXF",
    ],
    "lithium": [
        "ALB",
        "SQM",
        "LTHM",
        "PLL",
        "SGML",
        "LAC",
        "LIT",
        "ALTM",
        "OROCF",
        "LITOF",
        "PILBF",
        "LIACF",
        "GNENF",
        "CYDVF",
        "WMLSF",
        "LTUM",
        "PMETF",
        "IONR",
        "ATLX",
        "PGEZF",
        "STLHF",
        "ARREF",
        "AMLI",
        "CXOXF",
        "NRVTF",
        "MVDDF",
        "SXOOF",
        "ALLIF",
        "LLKKF",
        "BATT",
        "DRXLF",
        "PCELF",
        "MALRF",
        "GVXXF",
        "AMLM",
        "LILIF",
        "GALXF",
        "MGXRF",
        "MLNLF",
        "BATNF",
        "NBLXF",
        "SIOLF",
        "MAXRF",
        "QMCQF",
        "CRECF",
        "SBLGF",
        "LBRMF",
    ],
    "steel": [
        "NUE",
        "STLD",
        "CLF",
        "X",
        "AA",
        "CENX",
        "ATI",
        "RS",
        "CMC",
        "MT",
        "SID",
        "GGB",
        "TX",
        "SCHN",
        "TMST",
        "HAYN",
        "ZEUS",
        "MTRN",
        "WOR",
        "USAP",
        "SYNL",
        "IIIN",
        "NN",
        "SXC",
        "RYI",
        "PKX",
        "NISTF",
        "NSSMY",
        "TKAMY",
        "AZBHF",
        "CRS",
        "KALU",
        "CRD-A",
        "MTUS",
        "APOG",
        "STCN",
        "DOOR",
        "WIRE",
        "GBX",
        "TREC",
        "PRLB",
        "AZZ",
        "VMI",
        "MRC",
        "TITN",
    ],
    #  MATERIALS / CHEMICALS
    "materials": [
        "LIN",
        "APD",
        "SHW",
        "ECL",
        "DD",
        "DOW",
        "PPG",
        "VMC",
        "MLM",
        "CX",
        "CF",
        "MOS",
        "NTR",
        "FMC",
        "CTVA",
        "IIPR",
        "IP",
        "PKG",
        "GPK",
        "SEE",
        "SON",
        "BLL",
        "CCK",
        "EMN",
        "CE",
        "HUN",
        "TROX",
        "OLN",
        "KWR",
        "ASH",
        "CBT",
        "AXTA",
        "RPM",
        "MTX",
        "WDFC",
        "AMCR",
        "WRK",
        "BERY",
        "GEF",
        "ATR",
        "UFPI",
        "XYL",
        "WTRG",
        "SJW",
        "CWT",
        "AWK",
        "WMS",
        "FOE",
        "BCPC",
        "IOSP",
        "HWKN",
        "GCP",
        "NGVT",
    ],
    #  INDUSTRIALS
    "industrials": [
        "CAT",
        "DE",
        "HON",
        "GE",
        "GEV",
        "RTX",
        "LMT",
        "BA",
        "NOC",
        "GD",
        "LHX",
        "HII",
        "TDG",
        "HEI",
        "AXON",
        "TXT",
        "HWM",
        "CW",
        "SPR",
        "ETN",
        "EMR",
        "ROK",
        "AME",
        "NDSN",
        "ITW",
        "SWK",
        "IR",
        "PH",
        "DOV",
        "XYL",
        "ROP",
        "IEX",
        "GNRC",
        "TTC",
        "AGCO",
        "CNHI",
        "PCAR",
        "CMI",
        "OSK",
        "UPS",
        "FDX",
        "JBHT",
        "XPO",
        "ODFL",
        "SAIA",
        "KNX",
        "WERN",
        "LSTR",
        "CHRW",
        "DAL",
        "UAL",
        "LUV",
        "AAL",
        "ALK",
        "JBLU",
        "SKYW",
        "UNP",
        "CSX",
        "NSC",
        "CP",
        "CNI",
        "WM",
        "RSG",
        "CLH",
        "GFL",
        "CWST",
        "JCI",
        "CARR",
        "OTIS",
        "TT",
        "AOS",
        "MAS",
        "LII",
        "PWR",
        "TTEK",
        "MTZ",
        "DY",
        "FIX",
        "EMCOR",
        "HUBB",
        "AYI",
        "LECO",
        "ATKR",
        "FAST",
        "MSM",
        "GWW",
        "WCC",
        "POOL",
        "AVAV",
        "KTOS",
        "RKLB",
        "BKSY",
        "LUNR",
        "RDW",
        "ASTS",
        "WAB",
        "GWW",
        "RXO",
        "GATX",
    ],
    #  DEFENSE
    "defense": [
        "LMT",
        "RTX",
        "NOC",
        "GD",
        "BA",
        "LHX",
        "HII",
        "LDOS",
        "SAIC",
        "BAH",
        "KTOS",
        "AVAV",
        "MRCY",
        "CACI",
        "BWXT",
        "AXON",
        "SWBI",
        "RGR",
        "PSN",
        "KBR",
        "AMSC",
        "HXL",
        "CW",
        "TXT",
        "SPR",
        "TDG",
        "HEI",
        "HEI-A",
        "ESLT",
        "AJRD",
        "RKLB",
        "LUNR",
        "ASTS",
        "RDW",
        "BKSY",
        "JOBY",
        "LILM",
        "ACHR",
        "BLDE",
        "EVTL",
        "ALIT",
        "SEAH",
        "UFO",
        "ARKX",
        "ROKT",
    ],
    #  REAL ESTATE / REITs
    "real_estate": [
        "AMT",
        "PLD",
        "CCI",
        "EQIX",
        "PSA",
        "SPG",
        "O",
        "VICI",
        "DLR",
        "WELL",
        "EXR",
        "AVB",
        "EQR",
        "MAA",
        "UDR",
        "ESS",
        "CPT",
        "INVH",
        "AMH",
        "BXP",
        "VNO",
        "SLG",
        "KRC",
        "ARE",
        "OHI",
        "MPW",
        "SBRA",
        "STAG",
        "REXR",
        "FR",
        "TRNO",
        "NNN",
        "ADC",
        "EPRT",
        "IRM",
        "CUBE",
        "LSI",
        "NSA",
        "COLD",
        "GLPI",
        "RHP",
        "PK",
        "HST",
        "SUI",
        "ELS",
        "KRG",
        "REG",
        "FRT",
        "KIM",
        "IIPR",
        "GTY",
        "FCPT",
        "BNL",
        "LEN",
        "DHI",
        "PHM",
        "TOL",
        "KBH",
        "MDC",
        "MHO",
        "MTH",
        "GRBK",
        "TMHC",
        "CCS",
        "LGIH",
        "DFH",
    ],
    #  UTILITIES
    "utilities": [
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
        "EXC",
        "SRE",
        "XEL",
        "WEC",
        "ES",
        "ED",
        "DTE",
        "FE",
        "AEE",
        "CMS",
        "EVRG",
        "ATO",
        "NI",
        "PNW",
        "OGE",
        "PEG",
        "PPL",
        "CEG",
        "VST",
        "NRG",
        "AES",
        "AWK",
        "WTRG",
        "SJW",
        "CWT",
        "LNT",
        "NWN",
        "CNP",
        "BKH",
        "OTTR",
        "AVA",
        "MGEE",
        "UTL",
        "POR",
        "PCG",
        "EIX",
        "AGR",
        "IDA",
        "MDU",
        "OTTER",
        "MSEX",
        "ARTNA",
        "CWCO",
        "YORW",
        "SWX",
    ],
    #  COMMUNICATION SERVICES
    "communication": [
        "GOOGL",
        "META",
        "DIS",
        "NFLX",
        "CMCSA",
        "T",
        "VZ",
        "TMUS",
        "CHTR",
        "LBRDA",
        "FYBR",
        "LUMN",
        "SPOT",
        "ROKU",
        "ZM",
        "MTCH",
        "BMBL",
        "IAC",
        "RBLX",
        "TTWO",
        "EA",
        "WBD",
        "PARA",
        "FOXA",
        "NWSA",
        "NYT",
        "OMC",
        "IPG",
        "TTD",
        "MGNI",
        "DV",
        "GENI",
        "CARG",
        "ZD",
        "YELP",
        "SBAC",
        "SIRI",
        "LYV",
        "LBRDK",
        "ATUS",
        "CABO",
        "WOW",
        "CNSL",
        "GOGO",
        "IDT",
        "MSGS",
        "LGF-A",
        "LGF-B",
        "EDR",
        "WWE",
        "SGHC",
    ],
    #  AGRICULTURE
    "agriculture": [
        "ADM",
        "BG",
        "INGR",
        "DAR",
        "CALM",
        "FDP",
        "DE",
        "AGCO",
        "CNHI",
        "NTR",
        "MOS",
        "CF",
        "TSN",
        "HRL",
        "PPC",
        "ZTS",
        "IDXX",
        "LNN",
        "TITN",
        "SFM",
        "ANDE",
        "LMNR",
        "CTVA",
        "FMC",
        "IPI",
        "CORT",
        "VITL",
        "HYFM",
        "GRWG",
        "SMG",
        "SEED",
        "AVD",
        "AVO",
        "LIQT",
        "MOO",
        "DBA",
        "WEAT",
        "CORN",
        "SOYB",
        "TAGS",
        "FTXG",
        "VEGI",
        "COW",
        "NIB",
        "JO",
        "CANE",
        "SGG",
        "SOIL",
        "KROP",
    ],
    #  SHIPPING / MARITIME
    "shipping": [
        "ZIM",
        "MATX",
        "GOGL",
        "EGLE",
        "GNK",
        "SBLK",
        "STNG",
        "DAC",
        "NMM",
        "SFL",
        "CMRE",
        "GSL",
        "INSW",
        "NAT",
        "DHT",
        "FRO",
        "TRMD",
        "TK",
        "TGP",
        "DSX",
        "EXPD",
        "FWRD",
        "HUBG",
        "ARCB",
        "FLNG",
        "KNOP",
        "DLNG",
        "CPLP",
        "GASS",
        "TNK",
        "PSHG",
        "EDRY",
        "ESEA",
        "SHIP",
        "TOPS",
        "GLBS",
        "BWLP",
        "KEX",
        "SALT",
        "MPCC",
        "HAFN",
        "ASC",
        "CTRM",
        "SINO",
        "GTE",
        "PXS",
        "BDRY",
        "SEANF",
        "IMPP",
        "WAVE",
    ],
    #  CANNABIS
    "cannabis": [
        "TLRY",
        "CGC",
        "ACB",
        "CRON",
        "OGI",
        "SNDL",
        "HEXO",
        "VFF",
        "GRWG",
        "IIPR",
        "MSOS",
        "SMG",
        "TCNNF",
        "CURLF",
        "GTBIF",
        "TRSSF",
        "CRLBF",
        "VRNOF",
        "AYRWF",
        "CCHWF",
        "CLVRF",
        "PLNHF",
        "HBORF",
        "SHRM",
        "MRMD",
        "KERN",
        "FLGC",
        "HRVSF",
        "SSPK",
        "MAPS",
        "CBSTF",
        "GLASF",
        "AAWH",
        "FFNTF",
        "LHSIF",
        "CNTMF",
        "SPRTF",
        "ENTXF",
        "HITI",
        "CLSH",
        "NEPT",
        "GNLN",
        "SGMD",
        "PSDN",
    ],
    #  SOLAR
    "solar": [
        "ENPH",
        "FSLR",
        "SEDG",
        "RUN",
        "NOVA",
        "ARRY",
        "MAXN",
        "CSIQ",
        "JKS",
        "DQ",
        "SPWR",
        "SHLS",
        "PEGA",
        "HASI",
        "TAN",
        "ICLN",
        "FLNC",
        "EOSE",
        "BEEM",
        "ASTI",
        "OPTT",
        "SOL",
        "SOLARW",
        "SIRC",
        "SUNS",
        "NVEE",
        "ORA",
        "CWEN",
        "NEP",
        "BEP",
        "BEPC",
        "AY",
        "CSWI",
        "REGI",
        "GPRE",
        "LTBR",
        "ACES",
        "QCLN",
        "PBW",
        "FAN",
        "SMOG",
        "RAYS",
        "HYSR",
        "NRGV",
        "WATT",
        "SUNW",
    ],
    #  EV / ELECTRIC VEHICLES
    "ev": [
        "TSLA",
        "RIVN",
        "LCID",
        "NIO",
        "LI",
        "XPEV",
        "GOEV",
        "VFS",
        "JOBY",
        "LILM",
        "ACHR",
        "CHPT",
        "EVGO",
        "BLNK",
        "GM",
        "F",
        "STLA",
        "BWA",
        "APTV",
        "LEA",
        "QS",
        "MVST",
        "SLDP",
        "DCFC",
        "DRIV",
        "IDRV",
        "PTRA",
        "WKHS",
        "HYLN",
        "NKLA",
        "REE",
        "HYZN",
        "FREY",
        "FSR",
        "FFIE",
        "MULN",
        "ELMS",
        "RIDE",
        "KNDI",
        "AYRO",
        "CENN",
        "EVEX",
        "XPEV",
        "PSNY",
        "LCID",
        "SOLO",
        "ZEV",
        "VLCN",
    ],
    #  SPACE
    "space": [
        "RKLB",
        "SPCE",
        "LUNR",
        "ASTS",
        "RDW",
        "BKSY",
        "BA",
        "LMT",
        "NOC",
        "AVAV",
        "KTOS",
        "JOBY",
        "LILM",
        "AJRD",
        "HWM",
        "TDG",
        "HEI",
        "ESLT",
        "MAXR",
        "GSAT",
        "IRDM",
        "VSAT",
        "OUST",
        "INVZ",
        "ASTR",
        "MNTS",
        "VORB",
        "RKLA",
        "SATL",
        "PKE",
        "SWIR",
        "CXT",
        "GILT",
        "NGCA",
        "ACHR",
        "BLDE",
        "EVTL",
        "ALIT",
        "SEAH",
        "UFO",
        "ARKX",
        "ROKT",
    ],
    #  INTERNATIONAL ADRs
    "international": [
        "BABA",
        "PDD",
        "JD",
        "BIDU",
        "NIO",
        "LI",
        "XPEV",
        "NTES",
        "TME",
        "BILI",
        "ZTO",
        "YUMC",
        "MNSO",
        "FUTU",
        "TIGR",
        "VNET",
        "IQ",
        "YMM",
        "QFIN",
        "GDS",
        "WB",
        "TAL",
        "EDU",
        "TSM",
        "UMC",
        "ASX",
        "TM",
        "HMC",
        "SONY",
        "MUFG",
        "SMFG",
        "NMR",
        "ASML",
        "SAP",
        "NVO",
        "AZN",
        "GSK",
        "SNY",
        "SHOP",
        "RY",
        "TD",
        "BMO",
        "BNS",
        "CM",
        "CP",
        "CNI",
        "ENB",
        "TRP",
        "SU",
        "CNQ",
        "TECK",
        "NTR",
        "BCS",
        "HSBC",
        "DB",
        "UBS",
        "ING",
        "BBVA",
        "SAN",
        "BP",
        "SHEL",
        "TTE",
        "EQNR",
        "E",
        "DEO",
        "BUD",
        "MELI",
        "NU",
        "STNE",
        "PAGS",
        "GLOB",
        "SQM",
        "VALE",
        "PBR",
        "ITUB",
        "BBD",
        "ABEV",
        "BRFS",
        "SBS",
        "INFY",
        "WIT",
        "HDB",
        "IBN",
        "TTM",
        "RDY",
        "MMYT",
        "SE",
        "GRAB",
        "BHP",
        "RIO",
        "XRO",
        "GOLD",
        "SBSW",
        "HMY",
        "GFI",
        "DRD",
        "ERIC",
        "NOK",
        "STLA",
        "RACE",
        "LOMA",
        "GGAL",
        "PAM",
        "TGS",
        "CRH",
        "ICLR",
        "TEVA",
        "RELX",
        "WPP",
    ],
    #  CRYPTO-RELATED EQUITIES
    "crypto_equities": [
        "COIN",
        "MARA",
        "RIOT",
        "CLSK",
        "BTBT",
        "CIFR",
        "HUT",
        "BTDR",
        "CORZ",
        "WULF",
        "HOOD",
        "SQ",
        "MSTR",
        "ARBK",
        "BITF",
        "DMGGF",
        "HIVE",
        "BKKT",
        "SI",
        "BTCS",
        "BRPHF",
        "MOGO",
        "CBIT",
        "NUVB",
        "TAAL",
        "BFAM",
        "GREE",
        "IREN",
        "APLD",
        "DGHI",
        "CRPT",
        "BITQ",
        "DAPP",
        "BITO",
        "GBTC",
        "ETHE",
        "BLOK",
        "LEGR",
        "WGMI",
        "SATO",
        "IBIT",
        "FBTC",
        "ARKB",
        "HODL",
    ],
    #  SPECIAL SITUATIONS / MEMES
    "special": [
        "GME",
        "AMC",
        "CLOV",
        "WISH",
        "PLTR",
        "MVIS",
        "BB",
        "CLF",
        "RKT",
        "WKHS",
        "QS",
        "MVST",
        "DNA",
        "SPCE",
        "RKLB",
        "BKKT",
        "SOFI",
        "FUBO",
        "SKLZ",
        "DKNG",
        "CHPT",
        "LCID",
        "AI",
        "BBAI",
        "IONQ",
        "RGTI",
        "QUBT",
        "SOUN",
        "SMCI",
        "RIVN",
        "FFIE",
        "MULN",
        "NKLA",
        "GOEV",
        "ATER",
        "PROG",
        "IRNT",
        "DWAC",
        "PHUN",
        "MARK",
        "BBIG",
        "XELA",
        "GFAI",
        "RDBX",
        "APRN",
        "IMPP",
        "BRDS",
        "PRTY",
        "BGFV",
        "CENN",
    ],
}

# ETFs to scrape holdings from for dynamic expansion
_ETF_HOLDING_SOURCES = [
    "SPY",
    "QQQ",
    "IWM",
    "IWB",
    "IWV",  # broad US
    "VTI",
    "VTV",
    "VUG",
    "VBK",
    "VBR",  # Vanguard style
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",  # sectors
    "ARKK",
    "ARKQ",
    "ARKW",
    "ARKG",
    "ARKF",  # thematic
    "GDX",
    "GDXJ",
    "SIL",
    "COPX",
    "REMX",
    "URA",
    "LIT",  # commodities/mining
    "ICLN",
    "TAN",
    "QCLN",  # clean energy
    "HACK",
    "CIBR",
    "BUG",  # cyber
    "JETS",
    "ITA",  # airlines/defense
    "XBI",
    "IBB",  # biotech
    "KRE",
    "KBE",  # regional banks
    "ITB",
    "XHB",  # homebuilders
    "MSOS",  # cannabis
    "KWEB",
    "FXI",
    "EWJ",
    "EWZ",
    "EWY",
    "INDA",  # international
    "VEA",
    "VWO",
    "EEM",  # broad international
    "MDY",
    "IJH",  # mid-cap
    "IWN",
    "IWO",
    "IJR",  # small-cap
    "SOXX",
    "SMH",  # semis
    "XME",  # metals & mining
    "MOO",
    "DBA",  # agriculture
    "IYT",  # transports
    "PBW",  # clean energy
]

# Crypto tickers
_CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
    "SOL-USD",
    "DOGE-USD",
    "DOT-USD",
    "AVAX-USD",
    "SHIB-USD",
    "MATIC-USD",
    "LTC-USD",
    "ALGO-USD",
    "ATOM-USD",
    "LINK-USD",
    "XLM-USD",
    "VET-USD",
    "ICP-USD",
    "FIL-USD",
    "NEAR-USD",
    "APT-USD",
    "ARB-USD",
    "OP-USD",
    "SUI-USD",
    "HBAR-USD",
    "IMX-USD",
    "MANA-USD",
    "SAND-USD",
]

# DYNAMIC EXPANSION — fetch tickers from ETF holdings via yfinance
#


def _fetch_etf_holdings(etf_symbol: str, max_holdings: int = 100) -> List[str]:
    """Fetch top holdings of an ETF via yfinance."""
    try:
        import yfinance as yf

        tk = yf.Ticker(etf_symbol)

        # Method 1: .holdings or .get_holdings()
        holdings = None
        for attr in ["holdings", "top_holdings"]:
            h = getattr(tk, attr, None)
            if h is not None and hasattr(h, "__len__") and len(h) > 0:
                holdings = h
                break

        if holdings is not None and isinstance(
            holdings, (pd.DataFrame if "pd" in dir() else type(None))
        ):
            import pandas as pd

            if isinstance(holdings, pd.DataFrame):
                for col in ["Symbol", "symbol", "Ticker", "ticker", "holdingSymbol"]:
                    if col in holdings.columns:
                        return [
                            str(s).strip().upper()
                            for s in holdings[col].dropna().tolist()[:max_holdings]
                            if isinstance(s, str)
                            and 1 <= len(s.strip()) <= 6
                            and s.strip().isalpha()
                        ]

        # Method 2: fund_holding_info
        info = getattr(tk, "fund_holding_info", None)
        if info and isinstance(info, dict):
            equities = info.get("equityHoldings", {})
            if equities:
                return []  # no individual tickers here

        # Method 3: institutional_holders (not ETF holdings but works for some)
        return []
    except Exception:
        return []


def _fetch_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        import pandas as pd

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0
        )
        if tables:
            df = tables[0]
            for col in ["Symbol", "Ticker"]:
                if col in df.columns:
                    return [
                        str(s).strip().replace(".", "-")
                        for s in df[col].tolist()
                        if isinstance(s, str) and len(s.strip()) <= 6
                    ]
        return []
    except Exception:
        return []


def _fetch_russell2000_sample() -> List[str]:
    """Fetch a sample of Russell 2000 tickers."""
    try:
        import yfinance as yf

        # IWM top holdings give us some small-caps
        return _fetch_etf_holdings("IWM", max_holdings=200)
    except Exception:
        return []


def _fetch_nasdaq100_tickers() -> List[str]:
    """Fetch NASDAQ-100 tickers."""
    try:
        import pandas as pd

        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", header=0)
        for t in tables:
            for col in ["Ticker", "Symbol"]:
                if col in t.columns:
                    return [
                        str(s).strip()
                        for s in t[col].tolist()
                        if isinstance(s, str) and len(s.strip()) <= 6
                    ]
        return []
    except Exception:
        return []


def _validate_ticker(sym: str) -> bool:
    """Quick validation — just checks format, not data availability."""
    if not sym or not isinstance(sym, str):
        return False
    sym = sym.strip().upper()
    if len(sym) < 1 or len(sym) > 10:
        return False
    # Allow tickers like BF-B, BRK-B, BTC-USD, ES=F
    if any(c in sym for c in [" ", ",", ";", "(", ")", "[", "]", "{", "}"]):
        return False
    return True


#
# ASSET CLASS SEEDS
#

_CRYPTO_SEEDS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD",
    "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "LINK-USD", "BCH-USD", "SHIB-USD",
    "AVAX-USD", "ETC-USD", "XLM-USD", "XMR-USD", "UNI-USD", "ATOM-USD", "ALGO-USD"
]

_FOREX = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X", "EURCHF=X", "EURAUD=X",
    "EURCAD=X", "GBPAUD=X", "GBPCAD=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X"
]

_FUTURES = [
    "ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZB=F"
]


_ETFS = [
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO", "VGT", "XLK",
    "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE", "EEM", "FXI",
    "EWJ", "EWZ", "INDA", "ICLN", "HACK", "GDX", "GDXJ", "SIL", "COPX",
    "REMX", "URA", "LIT", "MSOS"
]


#
# UNIVERSE CLASS — main interface
#


class TickerUniverse:
    """Manages a 5000+ ticker universe with daily auto-refresh."""

    def __init__(self):
        self._stocks: Set[str] = set()
        self._sector_map: Dict[str, Set[str]] = {}
        self._etfs: Set[str] = set()
        self._crypto: Set[str] = set(_CRYPTO_SEEDS)
        self._forex: List[str] = list(_FOREX)
        self._futures: List[str] = list(_FUTURES)
        self._last_refresh: Optional[str] = None
        self._dynamic_added: Set[str] = set()  # tickers added dynamically
        self._load_or_build()

    def _load_or_build(self):
        """Load from cache if fresh (today), otherwise rebuild."""
        if self._try_load_cache():
            return
        self._build_from_seed()
        self._expand_dynamically()
        self._save_cache()

    def _try_load_cache(self) -> bool:
        """Try loading from daily cache file."""
        try:
            if not _CACHE_PATH.exists():
                return False
            with open(_CACHE_PATH, "r") as f:
                data = json.load(f)
            cached_date = data.get("date", "")
            today = datetime.now().strftime("%Y-%m-%d")
            if cached_date != today:
                return False
            self._stocks = set(data.get("stocks", []))
            self._sector_map = {
                k: set(v) for k, v in data.get("sector_map", {}).items()
            }
            self._etfs = set(data.get("etfs", []))
            self._crypto = set(data.get("crypto", []))
            self._forex = data.get("forex", list(_FOREX))
            self._futures = data.get("futures", list(_FUTURES))
            self._dynamic_added = set(data.get("dynamic_added", []))
            self._last_refresh = cached_date
            return len(self._stocks) > 100
        except Exception:
            return False

    def _save_cache(self):
        """Save current universe to daily cache."""
        try:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "stocks": sorted(self._stocks),
                "sector_map": {k: sorted(v) for k, v in self._sector_map.items()},
                "etfs": sorted(self._etfs),
                "crypto": sorted(self._crypto),
                "forex": self._forex,
                "futures": self._futures,
                "dynamic_added": sorted(self._dynamic_added),
                "total_count": len(self._stocks) + len(self._etfs) + len(self._crypto),
                "last_refresh_ts": datetime.now().isoformat(),
            }
            with open(_CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _build_from_seed(self):
        """Build universe from hardcoded seed lists + ComprehensiveUniverse."""
        self._stocks = set()
        self._sector_map = {}
        for sector, tickers in _SEED_BY_SECTOR.items():
            valid = {t.strip().upper() for t in tickers if _validate_ticker(t)}
            self._stocks.update(valid)
            self._sector_map[sector] = valid.copy()
        self._etfs = {t.strip().upper() for t in _ETFS if _validate_ticker(t)}
        self._crypto = set(_CRYPTO_SEEDS)

        # Bridge to ComprehensiveTickerUniverse for 10,000+ symbols
        try:
            from comprehensive_ticker_universe import ComprehensiveTickerUniverse
            ctu = ComprehensiveTickerUniverse()
            comp_stocks = ctu.get_stocks()
            self._stocks.update(comp_stocks)
            self._etfs.update(ctu.get_etfs())
            self._crypto.update(ctu.get_crypto())
            # Maintain uniqueness while merging
            self._forex = sorted(list(set(self._forex + ctu.get_forex())))
            self._futures = sorted(list(set(self._futures + ctu.get_futures())))
        except Exception as e:
            print(f"Warning: Could not bridge with ComprehensiveTickerUniverse: {e}")

    def _expand_dynamically(self):
        """Expand universe by fetching from external sources."""
        new_tickers: Set[str] = set()

        # 1. S&P 500
        try:
            sp500 = _fetch_sp500_tickers()
            new_tickers.update(sp500)
        except Exception:
            pass

        # 2. NASDAQ 100
        try:
            ndx = _fetch_nasdaq100_tickers()
            new_tickers.update(ndx)
        except Exception:
            pass

        # 3. ETF holdings (run in parallel for speed)
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = {
                    pool.submit(_fetch_etf_holdings, etf, 80): etf
                    for etf in _ETF_HOLDING_SOURCES[:30]
                }  # limit to avoid rate limits
                for fut in as_completed(futures, timeout=60):
                    try:
                        holdings = fut.result(timeout=10)
                        new_tickers.update(holdings)
                    except Exception:
                        pass
        except Exception:
            pass

        # 4. Russell 2000 sample
        try:
            r2k = _fetch_russell2000_sample()
            new_tickers.update(r2k)
        except Exception:
            pass

        # Filter and add
        valid_new = {
            t.strip().upper()
            for t in new_tickers
            if _validate_ticker(t) and t.strip().upper() not in self._stocks
        }
        self._stocks.update(valid_new)
        self._dynamic_added = valid_new

    def refresh_if_needed(self):
        """Check if a refresh is needed (new day) and rebuild if so."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_refresh == today:
            return
        with _LOCK:
            if self._last_refresh == today:
                return
            self._build_from_seed()
            self._expand_dynamically()
            self._last_refresh = today
            self._save_cache()

    #  PUBLIC API

    def get_all_stocks(self) -> List[str]:
        """Get all stock tickers (5000+)."""
        self.refresh_if_needed()
        return sorted(self._stocks)

    def get_sector_tickers(self, sector: str) -> List[str]:
        """Get tickers for a specific sector."""
        self.refresh_if_needed()
        return sorted(self._sector_map.get(sector, set()))

    def get_all_sectors(self) -> Dict[str, List[str]]:
        """Get all sector→tickers mapping."""
        self.refresh_if_needed()
        return {k: sorted(v) for k, v in self._sector_map.items()}

    def get_etfs(self) -> List[str]:
        self.refresh_if_needed()
        return sorted(self._etfs)

    def get_crypto(self) -> List[str]:
        return sorted(self._crypto)

    def get_forex(self) -> List[str]:
        return list(self._forex)

    def get_futures(self) -> List[str]:
        return list(self._futures)

    def get_all_tickers_flat(self) -> List[str]:
        """Get every ticker across all asset classes."""
        self.refresh_if_needed()
        all_t = set(self._stocks)
        all_t.update(self._etfs)
        all_t.update(self._crypto)
        return sorted(all_t)

    def get_symbol_sector(self, symbol: str) -> str:
        """Dynamically lookup the sector for any given symbol."""
        self.refresh_if_needed()
        sym = symbol.strip().upper()
        # Check standard sector map
        for sector, tickers in self._sector_map.items():
            if sym in tickers:
                return sector
        
        # Check if it's one of the other asset classes
        if sym in self._etfs: return "ETF"
        if sym in self._crypto: return "Crypto"
        if sym in self._forex: return "Forex"
        if sym in self._futures: return "Futures"
        
        return "Unknown"

    def get_all_tickers(self) -> List[str]:
        """Return every single ticker across ALL asset classes (Stocks, ETFs, Crypto, FX, Futures)."""
        self.refresh_if_needed()
        all_t = set(self._stocks)
        all_t.update(self._etfs)
        all_t.update(self._crypto)
        all_t.update(self._forex)
        all_t.update(self._futures)
        return sorted(all_t)

    def get_known_ticker_set(self) -> Set[str]:
        """Get a set of all known tickers for fast lookup."""
        self.refresh_if_needed()
        s = set(self._stocks)
        s.update(self._etfs)
        s.update(self._crypto)
        return s

    def get_random_sample(self, n: int = 200) -> List[str]:
        """Get a random unbiased sample of tickers."""
        self.refresh_if_needed()
        all_t = list(self._stocks)
        random.shuffle(all_t)
        return all_t[:n]

    def get_full_universe(self) -> List[str]:
        """Return the ENTIRE known universe (no sampling).

        Includes every stock, ETF, crypto, FX pair and futures contract the
        platform knows about — the union of the seed lists, the comprehensive
        universe bridge, and everything added dynamically (S&P 500, NASDAQ-100
        and ETF-holdings expansion). FX pairs are normalized to Yahoo "=X"
        format so they flow through the standard data pipeline.

        Used by scanners that must analyze ALL assets (e.g. Breaking Trades)
        instead of a fixed or sampled subset.
        """
        self.refresh_if_needed()

        def _yf_fx(pair: str) -> str:
            p = str(pair).strip().upper()
            if "/" in p and "=" not in p:
                return p.replace("/", "") + "=X"
            return p

        combined = (
            list(self._stocks)
            + list(self._etfs)
            + list(self._crypto)
            + [_yf_fx(p) for p in self._forex]
            + list(self._futures)
        )
        seen: set = set()
        deduped = []
        for t in combined:
            t = str(t).strip().upper()
            if t and t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    def get_full_universe_sample(self, n: int = 200) -> List[str]:
        """Get a random sample across ALL asset classes: stocks, ETFs, crypto, FX, futures.

        FX pairs are expressed in Yahoo Finance format (e.g. 'EURUSD=X') so they can be
        looked up through the standard data pipeline.  Futures and crypto are included with
        their native ticker format.
        """
        self.refresh_if_needed()

        # Stocks (largest bucket)
        stocks = list(self._stocks)
        random.shuffle(stocks)
        stock_quota = max(1, int(n * 0.70))

        # ETFs
        etfs = list(self._etfs)
        random.shuffle(etfs)
        etf_quota = max(1, int(n * 0.10))

        # Crypto
        crypto = list(self._crypto)
        random.shuffle(crypto)
        crypto_quota = max(1, int(n * 0.05))

        # FX — convert slash-format to Yahoo "=X" format for data pipeline compatibility
        # Ensure the five requested pairs are always present
        core_fx = [
            "USDJPY=X",
            "EURUSD=X",
            "EURCHF=X",
            "USDCAD=X",
            "EURAUD=X",
            "GBPUSD=X",
            "AUDUSD=X",
            "USDCHF=X",
            "NZDUSD=X",
            "EURJPY=X",
            "GBPJPY=X",
            "AUDJPY=X",
            "CADJPY=X",
            "USDTRY=X",
            "USDZAR=X",
            "USDMXN=X",
        ]
        fx_quota = max(1, int(n * 0.05))
        fx_sample = core_fx[:fx_quota]

        # Futures
        futures = list(self._futures)
        random.shuffle(futures)
        fut_quota = max(1, int(n * 0.10))

        combined = (
            stocks[:stock_quota]
            + etfs[:etf_quota]
            + crypto[:crypto_quota]
            + fx_sample
            + futures[:fut_quota]
        )

        # Deduplicate while preserving order, then shuffle and cap
        seen: set = set()
        deduped = []
        for t in combined:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        random.shuffle(deduped)
        return deduped[:n]

    # Extended FX pair list kept here for universe completeness
    CORE_FX_PAIRS = [
        "EUR/USD",
        "USD/JPY",
        "GBP/USD",
        "USD/CHF",
        "AUD/USD",
        "USD/CAD",
        "NZD/USD",
        "EUR/GBP",
        "EUR/JPY",
        "EUR/CHF",
        "EUR/AUD",
        "EUR/CAD",
        "EUR/NZD",
        "GBP/JPY",
        "GBP/CHF",
        "GBP/AUD",
        "GBP/CAD",
        "AUD/JPY",
        "AUD/CHF",
        "AUD/CAD",
        "AUD/NZD",
        "CHF/JPY",
        "CAD/JPY",
        "NZD/JPY",
        "USD/TRY",
        "USD/ZAR",
        "USD/MXN",
        "USD/HKD",
        "USD/SGD",
        "USD/NOK",
        "USD/SEK",
    ]

    def get_universe_stats(self) -> Dict[str, int]:
        """Get stats about the universe."""
        self.refresh_if_needed()
        return {
            "total_stocks": len(self._stocks),
            "total_etfs": len(self._etfs),
            "total_crypto": len(self._crypto),
            "total_forex": len(self._forex),
            "total_futures": len(self._futures),
            "total_sectors": len(self._sector_map),
            "dynamically_added": len(self._dynamic_added),
            "grand_total": len(self._stocks) + len(self._etfs) + len(self._crypto),
        }

    def add_ticker(self, symbol: str, sector: str = None):
        """Manually add a ticker to the universe."""
        sym = symbol.strip().upper()
        if _validate_ticker(sym):
            self._stocks.add(sym)
            if sector and sector in self._sector_map:
                self._sector_map[sector].add(sym)
            self._dynamic_added.add(sym)

    def get_sector_aliases(self) -> Dict[str, str]:
        """Return sector alias mapping."""
        return dict(_SECTOR_ALIASES)


# Sector aliases (same as chatbot uses)
_SECTOR_ALIASES = {
    "pharma": "healthcare",
    "pharmaceutical": "healthcare",
    "biotech": "healthcare",
    "bio": "healthcare",
    "medical": "healthcare",
    "health": "healthcare",
    "drug": "healthcare",
    "biopharma": "healthcare",
    "tech": "technology",
    "software": "technology",
    "semiconductor": "semiconductors",
    "chip": "semiconductors",
    "chips": "semiconductors",
    "semis": "semiconductors",
    "oil": "energy",
    "gas": "energy",
    "petroleum": "energy",
    "bank": "financials",
    "banking": "financials",
    "finance": "financials",
    "retail": "consumer",
    "shopping": "consumer",
    "discretionary": "consumer",
    "aerospace": "defense",
    "military": "defense",
    "weapons": "defense",
    "mine": "mining",
    "miner": "mining",
    "miners": "mining",
    "mineral": "minerals",
    "rare earth": "rare_earth",
    "rem": "rare_earth",
    "ree": "rare_earth",
    "critical minerals": "rare_earth",
    "rare earths": "rare_earth",
    "rare-earth": "rare_earth",
    "nuke": "uranium",
    "atomic": "uranium",
    "nuclear": "uranium",
    "farm": "agriculture",
    "farming": "agriculture",
    "ag": "agriculture",
    "agri": "agriculture",
    "food": "agriculture",
    "crop": "agriculture",
    "fertilizer": "agriculture",
    "potash": "agriculture",
    "weed": "cannabis",
    "marijuana": "cannabis",
    "pot": "cannabis",
    "boat": "shipping",
    "ship": "shipping",
    "freight": "shipping",
    "maritime": "shipping",
    "cyber": "cybersecurity",
    "security": "cybersecurity",
    "infosec": "cybersecurity",
    "rocket": "space",
    "satellite": "space",
    "orbit": "space",
    "photovoltaic": "solar",
    "pv": "solar",
    "electric vehicle": "ev",
    "electric vehicles": "ev",
    "evs": "ev",
    "green energy": "clean_energy",
    "renewable": "clean_energy",
    "renewables": "clean_energy",
    "green": "clean_energy",
    "wind": "clean_energy",
    "hydrogen": "clean_energy",
    "precious metals": "gold",
    "bullion": "gold",
    "iron": "steel",
    "iron ore": "steel",
    "steelmaker": "steel",
    "battery": "lithium",
    "batteries": "lithium",
    "li-ion": "lithium",
    "telecom": "communication",
    "media": "communication",
    "reit": "real_estate",
    "reits": "real_estate",
    "property": "real_estate",
    "staples": "consumer_staples",
    "household": "consumer_staples",
    "crypto": "crypto_equities",
    "bitcoin": "crypto_equities",
    # Commodities (maps to energy + mining + agriculture for broad coverage)
    "commodity": "energy",
    "commodities": "energy",
    "raw materials": "mining",
    "precious metals": "gold",
    "base metals": "mining",
    "grains": "agriculture",
    "softs": "agriculture",
}


#  SINGLETON


def get_ticker_universe() -> TickerUniverse:
    """Get the singleton ticker universe instance."""
    global _universe_instance
    if _universe_instance is None:
        with _LOCK:
            if _universe_instance is None:
                _universe_instance = TickerUniverse()
    return _universe_instance
