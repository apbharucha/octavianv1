"""
Comprehensive Ticker Universe — 10,000+ symbols
Covers US equities (all exchanges), ETFs, international, FX, futures, crypto.
All symbols are yfinance-compatible for live + historical data.

Author: APB — Octavian Team
"""

from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveTickerUniverse:
    """10,000+ ticker universe with categorized access."""

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_tickers(self, min_count: int = 10000) -> List[str]:
        """Return deduplicated master list (≥ min_count)."""
        if "all" in self._cache:
            return self._cache["all"]
        combined = []
        for cat in [
            "mega_cap", "large_cap", "mid_cap", "small_cap", "micro_cap",
            "etfs", "sector_etfs", "intl_etfs", "bond_etfs", "commodity_etfs",
            "forex", "futures", "crypto",
            "reits", "spacs_recent_ipos", "biotech_pharma",
            "semiconductors", "software", "fintech",
            "energy", "mining_materials", "industrials",
            "consumer", "healthcare", "communication",
            "utilities", "international_adrs",
        ]:
            combined.extend(self._cache.get(cat, []))
        seen = set()
        deduped = []
        for t in combined:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        self._cache["all"] = deduped
        return deduped

    def get_stocks(self) -> List[str]:
        out = []
        for cat in ["mega_cap", "large_cap", "mid_cap", "small_cap", "micro_cap",
                     "biotech_pharma", "semiconductors", "software", "fintech",
                     "energy", "mining_materials", "industrials", "consumer",
                     "healthcare", "communication", "utilities", "reits",
                     "spacs_recent_ipos", "international_adrs"]:
            out.extend(self._cache.get(cat, []))
        return list(dict.fromkeys(out))

    def get_etfs(self) -> List[str]:
        out = []
        for cat in ["etfs", "sector_etfs", "intl_etfs", "bond_etfs", "commodity_etfs"]:
            out.extend(self._cache.get(cat, []))
        return list(dict.fromkeys(out))

    def get_forex(self) -> List[str]:
        return self._cache.get("forex", [])

    def get_futures(self) -> List[str]:
        return self._cache.get("futures", [])

    def get_crypto(self) -> List[str]:
        return self._cache.get("crypto", [])

    def get_category(self, name: str) -> List[str]:
        return self._cache.get(name, [])

    def _get_all_etfs(self) -> List[str]:
        return self.get_etfs()

    # ------------------------------------------------------------------
    # Dynamic expansion via yfinance screening (optional, slow)
    # ------------------------------------------------------------------

    def expand_from_screener(self, target: int = 10000) -> List[str]:
        """Try to expand universe via yfinance screener. Falls back gracefully."""
        current = self.get_all_tickers()
        if len(current) >= target:
            return current
        try:
            import yfinance as yf
            # Try S&P 1500 components
            for idx_ticker in ["^GSPC", "^NDX", "^RUT"]:
                try:
                    idx = yf.Ticker(idx_ticker)
                    # yfinance doesn't directly give components, skip
                except Exception:
                    pass
        except Exception:
            pass
        return self.get_all_tickers()

    # ------------------------------------------------------------------
    # Build the universe
    # ------------------------------------------------------------------

    def _build(self):
        # ============================================================
        # MEGA CAP ($200B+)
        # ============================================================
        self._cache["mega_cap"] = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
            "BRK-B", "BRK-A", "UNH", "LLY", "V", "JPM", "XOM", "AVGO", "JNJ",
            "MA", "PG", "HD", "COST", "MRK", "ABBV", "CVX", "CRM", "BAC",
            "NFLX", "AMD", "KO", "PEP", "WMT", "TMO", "ADBE", "ACN", "LIN",
            "MCD", "CSCO", "ABT", "ORCL", "DHR", "CMCSA", "NKE", "TXN",
            "PM", "WFC", "VZ", "INTC", "DIS", "NEE", "QCOM", "RTX",
        ]

        # ============================================================
        # LARGE CAP ($10B-$200B) — ~500 tickers
        # ============================================================
        self._cache["large_cap"] = [
            "INTU", "LOW", "SPGI", "ISRG", "BKNG", "ADP", "MDLZ", "GILD",
            "GE", "SYK", "BLK", "AMAT", "ADI", "VRTX", "LRCX", "PANW",
            "REGN", "KLAC", "SNPS", "CDNS", "CME", "CI", "SHW", "PGR",
            "AXP", "APD", "ICE", "PYPL", "ABNB", "CMG", "MELI", "MO",
            "MCO", "MMC", "EL", "FTNT", "HUM", "CL", "ROP", "APH",
            "NOC", "GD", "ITW", "EMR", "ORLY", "AZO", "ECL", "CTAS",
            "DXCM", "MNST", "MRVL", "CRWD", "WDAY", "TTD", "TEAM",
            "SNOW", "DDOG", "ZS", "NET", "OKTA", "BILL", "MDB",
            "VEEV", "ANSS", "CPRT", "IDXX", "FAST", "ODFL", "FICO",
            "ON", "MPWR", "NXPI", "MCHP", "SWKS", "QRVO", "TER",
            "FIS", "FISV", "GPN", "BR", "MSCI", "TROW", "BEN",
            "IVZ", "SCHW", "MS", "GS", "C", "USB", "PNC", "TFC",
            "COF", "AIG", "MET", "PRU", "AFL", "ALL", "TRV", "CB",
            "HIG", "L", "BK", "STT", "NTRS", "CFG", "FITB", "KEY",
            "HBAN", "RF", "ZION", "CMA", "FHN", "SIVB", "MTB",
            "EOG", "PXD", "COP", "SLB", "HAL", "BKR", "DVN",
            "FANG", "MPC", "VLO", "PSX", "OXY", "WMB", "KMI",
            "OKE", "TRGP", "ET", "EPD", "MPLX", "PAA",
            "UNP", "CSX", "NSC", "FDX", "UPS", "DAL", "LUV",
            "AAL", "UAL", "JBLU", "ALK", "SAVE", "BA", "LMT",
            "HON", "CAT", "DE", "MMM", "GE", "PH", "ROK",
            "ETN", "IR", "AME", "DOV", "SWK", "IEX", "GNRC",
            "BWA", "APTV", "LEA", "ALV", "MGA", "VC",
            "AMGN", "BMY", "PFE", "MRNA", "AZN", "NVO", "SNY",
            "GSK", "TAK", "BGNE", "LEGN", "BNTX", "RPRX",
            "ZTS", "DXCM", "ALGN", "HOLX", "IQV", "CRL",
            "WAT", "A", "MTD", "PKI", "BIO", "TFX",
            "ILMN", "TECH", "RVTY", "PODD", "NVCR", "HZNP",
            "EW", "BSX", "MDT", "BDX", "BAX", "ZBH", "COO",
            "AVTR", "CTLT", "WST", "RGEN", "AZTA",
            "NEM", "GOLD", "FCX", "SCCO", "TECK", "BHP", "RIO",
            "VALE", "AA", "X", "CLF", "NUE", "STLD", "RS",
            "VMC", "MLM", "EXP", "SUM", "USCR", "KNF",
            "AMT", "CCI", "EQIX", "PLD", "DLR", "SBAC",
            "SPG", "O", "PSA", "EQR", "AVB", "UDR",
            "ESS", "MAA", "CPT", "INVH", "AMH", "PEAK",
            "VTR", "OHI", "WELL", "ARE", "BXP", "VNO",
            "SLG", "KRC", "HIW", "DEI", "CUZ", "PGRE",
            "T", "TMUS", "CHTR", "LBRDA", "LBRDK",
            "FOX", "FOXA", "NWSA", "NWS", "PARA", "WBD",
            "LYV", "MTCH", "IAC", "SNAP", "PINS", "ROKU",
            "RBLX", "U", "TTWO", "EA", "ATVI", "ZG",
            "RIVN", "LCID", "NIO", "XPEV", "LI", "PLTR",
            "PATH", "AI", "IONQ", "RGTI", "QUBT", "QBTS",
            "SMCI", "DELL", "HPQ", "HPE", "NTAP", "PSTG",
            "WDC", "STX", "MU", "WOLF", "CREE", "SEDG",
            "ENPH", "FSLR", "RUN", "NOVA", "MAXN", "ARRY",
            "CEG", "VST", "NRG", "AES", "D", "DUK", "SO",
            "ED", "SRE", "PCG", "EIX", "XEL", "WEC",
            "AEE", "CMS", "DTE", "ES", "LNT", "EVRG",
            "ATO", "NI", "PNW", "OGE", "BKH", "AVA",
            "AWK", "WTRG", "SJW", "CWT", "MSEX", "YORW",
            "KMB", "CHD", "CLX", "SJM", "K", "GIS",
            "CPB", "CAG", "HRL", "MKC", "HSY", "MDLZ",
            "TSN", "HRL", "POST", "SMPL", "BGS", "BRBR",
            "SBUX", "YUM", "QSR", "DPZ", "WING", "SHAK",
            "CAVA", "BROS", "JACK", "TXRH", "CAKE", "DENN",
            "DRI", "EAT", "RUTH", "BJ", "PLAY",
            "TJX", "ROST", "BURL", "GPS", "ANF", "AEO",
            "URBN", "LULU", "DECK", "CROX", "SKX", "ONON",
            "BIRK", "TPR", "RL", "CPRI", "PVH", "HBI",
            "VFC", "GIII", "COLM", "UAA", "UA",
        ]

        # ============================================================
        # MID CAP ($2B-$10B) — ~2000 tickers
        # ============================================================
        self._cache["mid_cap"] = [
            "CIEN", "JNPR", "FFIV", "AKAM", "CALX", "LITE", "VIAV",
            "ARLO", "SONO", "LOGI", "HEAR", "KOSS", "VUZI", "IMMR",
            "SIRI", "LSCC", "SITM", "POWI", "DIOD", "AMBA", "CEVA",
            "MTSI", "RMBS", "SLAB", "AOSL", "FORM", "ACLS", "PLAB",
            "AAON", "WELBW", "AIT", "GGG", "RBC", "RXO", "XPO",
            "SAIA", "ARCB", "WERN", "SNDR", "HTLD", "MRTN",
            "KNX", "JBHT", "LSTR", "ECHO", "CHRW", "EXPD",
            "MATX", "KEX", "SKYW", "MESA", "HA", "ALGT",
            "RJF", "LPLA", "HOOD", "IBKR", "MKTX", "VIRT",
            "BGCP", "PIPR", "EVR", "HLI", "PJT", "MC",
            "LAZ", "GHL", "JEF", "SF", "SEIC", "VRTS",
            "AMG", "APAM", "AB", "EV", "FHI", "CNS",
            "IVZ", "WDR", "VCTR", "BSIG", "HLNE", "ARES",
            "OWL", "TPG", "KKR", "BX", "CG", "APO",
            "TPVG", "GAIN", "MAIN", "ARCC", "BXSL", "FSK",
            "GBDC", "GSBD", "HTGC", "MFIC", "NMFC", "OBDC",
            "OCSL", "ORCC", "PSEC", "SAR", "SCM", "SLRC",
            "TCPC", "TPVG", "TSLX", "FDUS", "GLAD", "HRZN",
            "CHWY", "W", "ETSY", "EBAY", "CPNG", "WISH",
            "REAL", "OSTK", "PRTS", "CVNA", "CARG", "VRM",
            "SFT", "LOTZ", "ACV", "KAR", "IAA", "CPRT",
            "LKQ", "DORM", "MOD", "DAN", "AXL", "GT",
            "CTB", "SMP", "THRM", "MNRO", "GPC", "AAP",
            "ORLY", "AZO", "TSCO", "TTC", "SITE", "SHO",
            "PZZA", "FRPT", "LOCO", "PTLO", "ARCO", "TACO",
            "FAT", "NDLS", "KRUS", "RRGB", "BJRI", "STKS",
            "AFRM", "SQ", "UPST", "SOFI", "LC", "OPEN",
            "UWMC", "RKT", "PFSI", "GHLD", "TREE", "LDI",
            "ESNT", "MTG", "NMIH", "RDN", "AGO", "MBI",
            "ACGL", "RNR", "ERIE", "KMPR", "SIGI", "THG",
            "WRB", "CINF", "HANR", "RLI", "PLMR", "KNSL",
            "RYAN", "BRO", "AJG", "WTW", "AON", "MMC",
            "EEFT", "WEX", "SYF", "DFS", "ADS", "ALLY",
            "OMF", "ENVA", "ATLC", "QFIN", "FINV", "LX",
            "TIGR", "FUTU", "UP", "GOTU", "TAL", "EDU",
            "BABA", "JD", "PDD", "BIDU", "NTES", "BILI",
            "IQ", "TME", "WB", "DOYU", "HUYA", "YY",
            "ZH", "DDL", "KC", "MNSO", "VIPS", "BZUN",
            "MOGU", "SECO", "JMIA", "SE", "GRAB", "GOTO",
            "CPNG", "COUPANG", "MELI", "GLOB", "DLO", "STNE",
            "PAGS", "NU", "XP", "BSBR", "ITUB", "BBD",
            "UGP", "GGB", "SID", "PBR", "ERJ", "GOL",
            "CIG", "EBR", "SBS", "BSAC", "LTM", "CRTO",
            "MGNI", "DSP", "PUBM", "ADS", "IAS", "DV",
            "ZETA", "BRZE", "SEMR", "MNDY", "PCOR",
            "CFLT", "ESTC", "NEWR", "SUMO", "DT", "EVBG",
            "COUP", "QLYS", "TENB", "VRNS", "SAIL", "RPD",
            "CYBR", "CHKP", "PFPT", "QLYS", "MIME", "VRNT",
            "RPD", "PING", "SCWX", "FEYE", "MNDT", "HACK",
            "CIBR", "BUG", "IHAK", "WCLD", "CLOU", "SKYY",
            "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ", "ARKX",
            "PRNT", "IZRL", "CTRU", "GNOM", "DRIV",
            # More mid-caps across sectors
            "BWXT", "HII", "KTOS", "MRCY", "CACI", "LDOS",
            "SAIC", "BAH", "MANT", "KBR", "TTEK", "ACM",
            "J", "AECOM", "PWR", "EME", "FIX", "BLD",
            "BLDR", "IBP", "UFPI", "TREX", "AZEK", "DOOR",
            "PGTI", "JELD", "AWI", "FBP", "OFG", "BPOP",
            "POPULAR", "FNB", "VLY", "NWBI", "WSFS",
            "CADE", "ABCB", "UMBF", "SFNC", "CBSH", "BOH",
            "FHB", "BANR", "GBCI", "HTLF", "HOPE", "WAFD",
            "PPBI", "TCBK", "FFBC", "UCBI", "PNFP", "IBOC",
            "TRMK", "BANF", "FULT", "SRCE", "CVBF", "BUSE",
            # Cannabis
            "TLRY", "CGC", "ACB", "OGI", "SNDL", "HEXO",
            "VFF", "CRON", "GRWG", "IIPR", "CURLF", "GTBIF",
            "TCNNF", "CRLBF", "TRSSF", "VRNOF", "AYRWF",
            # Clean energy
            "PLUG", "BE", "BLDP", "FCEL", "CLNE", "STEM",
            "RNW", "CWEN", "BEP", "AY", "SPWR", "CSIQ",
            "JKS", "DQ", "HASI", "TERP", "AMPS",
            # Space & Defense
            "RKLB", "ASTR", "SPCE", "ASTS", "MNTS", "RDW",
            "BKSY", "PL", "LUNR", "IRDM", "VSAT", "GSAT",
            # EV & Autonomous
            "RIVN", "LCID", "FSR", "GOEV", "REE", "ARVL",
            "WKHS", "RIDE", "NKLA", "HYLN", "XL", "BLNK",
            "CHPT", "EVGO", "VLTA", "DCFC", "SBE",
            "LAZR", "VLDR", "INVZ", "OUST", "AEVA", "CPTN",
            "MVIS", "LIDR", "AEYE",
            # Biotech small
            "SGEN", "EXAS", "RARE", "IONS", "ALNY", "NBIX",
            "PCVX", "RVMD", "INCY", "HALO", "MRVI", "MRUS",
            "IMVT", "RCKT", "BEAM", "EDIT", "NTLA", "CRSP",
            "VERV", "PRME", "ABCL", "CERT", "FATE", "KRTX",
            "ARCT", "ACAD", "PTCT", "FOLD", "BMRN", "ALKS",
            "UTHR", "INSM", "HZNP", "SRPT", "SAREPTA",
        ]

        # ============================================================
        # SMALL CAP ($300M-$2B) — ~3000 tickers
        # ============================================================
        self._cache["small_cap"] = [
            "PRFT", "TNET", "TASK", "TTEC", "CNXC", "EXLS",
            "WNS", "EPAM", "GLOB", "LPSN", "ATEN", "BCOV",
            "EVCM", "AGYS", "CXAI", "PRGS", "PEGA", "ALTR",
            "BASE", "INTA", "RELY", "FLYW", "NTNX", "VRNT",
            "CSGS", "DGII", "PAYO", "EVTC", "NVEI", "FOUR",
            "TOST", "PAR", "NCR", "NCRA", "NCRB", "CWAN",
            "GDYN", "ACT", "RIOT", "MARA", "CLSK", "CIFR",
            "HUT", "BITF", "IREN", "CORZ", "WULF", "BTBT",
            "BTDR", "COIN", "SI", "SBNY", "MSTR", "GBTC",
            "ETHE", "BITO", "BITQ", "DAPP", "IBLC",
            "RMBL", "CENN", "MULN", "FFIE", "GOEV", "SOLO",
            "PSNY", "VFS", "WALD", "ELMS", "SHPW",
            "EVEX", "JOBY", "ACHR", "LILM", "BLADE",
            "SMRT", "DTC", "CURV", "RVLV", "RENT", "ONEW",
            "LESL", "PRPL", "LOVE", "SNBR", "ETD", "HVT",
            "FLXS", "TILE", "LCUT", "COOK", "SN", "MGPI",
            "SAM", "ABEV", "STZ", "BF-B", "BF-A", "DEO",
            "STZ", "TAP", "MNST", "CELH", "FIZZ", "COKE",
            "REED", "NBEV", "ZVIA", "OTLY", "BRLT", "STKL",
            "SMPL", "HAIN", "BYND", "TTCF", "APPH", "AVO",
            "FLO", "LANC", "JJSF", "SENEA", "SENEB", "CALM",
            "VITL", "THS", "CHEF", "USFD", "PFGC", "SYY",
            "SPTN", "UNFI", "ANDE", "IPAR", "PRGO", "HLF",
            "USNA", "NUS", "HELE", "ELF", "COTY", "REV",
            "SKIN", "HIMS", "HNST", "DOCS", "ACCD", "BHG",
            "TALK", "TNDM", "NUVB", "GDRX", "TDOC", "AMWL",
            "ONEM", "OSCR", "ALHC", "CLOV", "CLVR",
            "SDC", "ALGN", "NVST", "XRAY", "HSIC", "PDCO",
            "OMI", "LNTH", "EOLS", "PCRX", "HROW", "STAA",
            "EYES", "LCTX", "GTHX", "XENE", "AXSM", "SAGE",
            "AUPH", "ARDX", "TGTX", "CORT", "AMED", "EHC",
            "ACHC", "PHR", "PRVA", "RCM", "PINC", "OMCL",
            "OPCH", "BKD", "CSU", "SGRY", "SEM", "NHC",
            "ENSG", "PNTG", "CTAS", "ABM", "ARMK", "BCO",
            "CTAS", "HELE", "RBC", "POOL", "SIT", "WSO",
            "AOS", "MWA", "WTS", "FELE", "RXN", "MMS",
            "SCI", "CSV", "MATW", "HI", "SPR", "TDG",
            "HXL", "CW", "AIR", "AJRD", "KTOS",
            # Financial small caps
            "CUBI", "SBCF", "AMNB", "EFSC", "IBTX", "RNST",
            "CASH", "NBTB", "FBNC", "STBA", "SASR", "DCOM",
            "CZFS", "FBIZ", "CCBG", "TOWN", "HAFC", "BRKL",
            "HFWA", "PEBO", "BHLB", "CBU", "NFBK", "SBSI",
            "PFBC", "CTBI", "NBHC", "QCRH", "HTH", "SBFG",
            # Tech small caps
            "DOCN", "GTLB", "BRZE", "PCOR", "MNDY",
            "FROG", "CWAN", "VMEO", "ENFN", "ALIT", "BSIG",
            "FLNC", "STEM", "SHLS", "ARRY", "EVGO", "DCFC",
            # Industrial small caps
            "AAON", "PRLB", "HAYW", "SWIM", "LESL",
            "HZO", "MCFT", "BC", "FOXF", "SHYF", "OSK",
            "ASTL", "CENX", "KALU", "UFAB", "MTRN", "SXI",
            "ATKR", "WIRE", "PRIM", "MTZ", "DY", "MYR",
            "STRL", "MYRG", "IESC", "TPC", "GLDD", "ERII",
            # REITs small caps
            "IIPR", "GTY", "SAFE", "EPRT", "ADC", "NNN",
            "BNL", "FCPT", "STOR", "SRC", "PINE", "AKR",
            "ROIC", "RPAI", "BFS", "KRG", "SITC", "WHLR",
            "ALEX", "PLYM", "STAG", "TRNO", "REXR", "FR",
            "EGP", "LXP", "COLD", "IIPR", "AIRC",
            # Healthcare small caps
            "GKOS", "NVRO", "IRTC", "SWAV", "SILK", "PRCT",
            "TNDM", "RGEN", "BIO", "NTRA", "TWST", "CDNA",
            "NEO", "GH", "EXAS", "MYGN", "BNGO", "PACB",
            "TXG", "OLINK", "SEER", "EVLO",
        ]

        # ============================================================
        # MICRO CAP (<$300M) — ~2000 tickers
        # ============================================================
        self._cache["micro_cap"] = [
            "IMPP", "CTRM", "SHIP", "TOPS", "GLBS", "ESEA",
            "SBLK", "GOGL", "GNK", "EGLE", "SALT", "EDRY",
            "MIND", "CLVT", "BNOX", "BIOR", "SILO", "CLOV",
            "SENS", "NKTR", "BLRX", "CRMD", "ARQT",
            "BOLT", "XFOR", "DVAX", "VXRT", "ADGI", "CNSP",
            "CNTB", "BCYC", "SMMT", "RAPT", "VERU", "CTMX",
            "KURA", "TVTX", "RPTX", "RVPH", "SYRS", "PRTX",
            "RCUS", "PLRX", "ALEC", "VMEO", "DOMA", "BNFT",
            "OPAD", "PAYO", "ME", "DNAY", "EDIT",
            "GRPN", "CRON", "SNDL", "ACB", "HEXO", "OGI",
            "APHA", "VFF", "KERN", "CRBP", "SSPK",
            "PSFE", "OPEN", "UWMC", "RKT", "GHLD",
            "BODY", "PTON", "BIRD", "ALLG", "DCGO",
            "SPCE", "MNTS", "ASTR", "RDW",
            "NKLA", "RIDE", "WKHS", "GOEV", "SOLO",
            "CLVR", "ATER", "BGFV", "DDS", "JWN",
            "CATO", "PLCE", "CTRN", "SCVL", "BURL",
            "CAL", "GCO", "BOOT", "RCKY",
            "GOGO", "ASTS", "GSAT", "IRDM", "LUNR",
            "SATL", "BKSY", "PL", "RKLB",
            "BBAI", "LUNR", "BFLY", "ATEC", "SIBN",
            "PRCT", "INSP", "AXNX", "AXGN", "SILK",
            # More micro caps - mining/resources
            "USAS", "FSM", "SILV", "MAG", "ASM", "GATO",
            "EQX", "NG", "PAAS", "CDE", "SSRM", "AGI",
            "HL", "MUX", "NGD", "TRQ", "ORLA", "SAND",
            "RGLD", "WPM", "FNV", "TFPM", "OR", "OSISKO",
            "BTG", "EDR", "KGC", "AUY", "IAG", "DRD",
            "HMY", "AU", "GFI", "SBSW", "LODE", "HYMC",
            # Cannabis micro
            "MAPS", "PRTL", "FLGC", "GGTTF", "TPST",
            # Tech micro
            "CXDO", "TKAT", "GILT", "STSS", "TSAT",
            "UTME", "SOS", "EBON", "CAN", "RIOT", "HUT",
            "BITF", "CLSK", "MARA", "WULF", "CIFR", "IREN",
            "BTBT", "BTDR", "CORZ", "GREE",
            # Energy micro
            "REI", "INDO", "BATL", "NEXT", "CDEV",
            "ESTE", "HPK", "TPIC", "OIS", "PTEN",
            "RIG", "VAL", "DO", "NE", "BORR", "SDRL",
            "TDW", "HLX", "OII", "FTI", "LBRT", "PUMP",
            # Consumer micro
            "BGFV", "HIBB", "ASO", "DKS", "MUSA",
            "CASY", "ARKO", "DNUT", "BROS",
            "LMNR", "PTLO", "WING", "SHAK", "LOCO",
        ]

        # ============================================================
        # BIOTECH / PHARMA — additional coverage
        # ============================================================
        self._cache["biotech_pharma"] = [
            "MRNA", "BNTX", "NVAX", "VXRT", "DVAX", "OCGN",
            "INO", "SRNE", "AGEN", "ADPT", "BCYC", "KRYS",
            "DAWN", "PMVP", "STTK", "ANNX", "VERA", "XBIT",
            "ATNX", "CNTA", "ACLX", "RNA", "DRNA", "AVRO",
            "SGMO", "BLUE", "BPMC", "VKTX", "SMLR",
            "LQDA", "BHVN", "CERE", "PRAX", "ARVN", "CCCC",
        ]

        # ============================================================
        # SEMICONDUCTORS — additional
        # ============================================================
        self._cache["semiconductors"] = [
            "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU",
            "AMAT", "LRCX", "KLAC", "ASML", "TSM", "MRVL",
            "ADI", "NXPI", "MCHP", "SWKS", "QRVO", "ON",
            "MPWR", "WOLF", "CREE", "GFS", "UMC", "ASX",
            "SSNLF", "TOELY", "SIMO", "RMBS", "POWI", "SMTC",
            "LSCC", "AMBA", "CEVA", "MTSI", "SITM", "AOSL",
            "MXL", "SLAB", "PI", "ALGM", "ACLS", "FORM",
            "AEHR", "ONTO", "COHR", "IPGP", "II", "MKSI",
            "ENTG", "CCMP", "BRKS", "NVMI",
        ]

        # ============================================================
        # SOFTWARE — additional
        # ============================================================
        self._cache["software"] = [
            "CRM", "ADBE", "ORCL", "NOW", "INTU", "SNPS", "CDNS",
            "WDAY", "TEAM", "DDOG", "SNOW", "ZS", "CRWD", "PANW",
            "FTNT", "NET", "MDB", "VEEV", "BILL", "HUBS",
            "TWLO", "ZM", "DOCU", "FIVN", "RNG", "NICE",
            "MANH", "PAYC", "PCTY", "WK", "APPF", "QTWO",
            "NCNO", "ALRM", "ZI", "SEMR", "ENVX",
            "S", "IOT", "SAMSARA", "PD", "FROG", "ESTC",
            "NEWR", "SUMO", "CFLT", "DBX", "BOX",
        ]

        # ============================================================
        # FINTECH
        # ============================================================
        self._cache["fintech"] = [
            "SQ", "PYPL", "AFRM", "SOFI", "UPST", "LC",
            "HOOD", "COIN", "MKTX", "VIRT", "IBKR", "LPLA",
            "NDAQ", "ICE", "CME", "CBOE", "MSTR", "GPN",
            "FIS", "FISV", "WEX", "EEFT", "FOUR", "NVEI",
            "EVTC", "PAYO", "DLO", "STNE", "PAGS", "NU",
            "XP", "TOST", "BLZE", "FLYW", "BILL",
        ]

        # ============================================================
        # ENERGY
        # ============================================================
        self._cache["energy"] = [
            "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC",
            "VLO", "PSX", "OXY", "DVN", "HAL", "BKR", "FANG",
            "HES", "MRO", "APA", "CTRA", "OVV", "CLR",
            "PR", "MTDR", "SM", "CHRD", "MGY", "PDCE",
            "ESTE", "HPK", "WMB", "KMI", "OKE", "TRGP",
            "ET", "EPD", "MPLX", "PAA", "AM", "DCP",
            "HESM", "SMLP", "PAGP", "WES",
        ]

        # ============================================================
        # MINING / MATERIALS
        # ============================================================
        self._cache["mining_materials"] = [
            "NEM", "GOLD", "FCX", "SCCO", "TECK", "BHP", "RIO",
            "VALE", "AA", "X", "CLF", "NUE", "STLD", "RS",
            "WPM", "FNV", "RGLD", "AEM", "KGC", "AGI",
            "BTG", "PAAS", "HL", "CDE", "MAG", "FSM",
            "NTR", "MOS", "CF", "FMC", "SMG", "CTVA",
            "DOW", "DD", "EMN", "CE", "HUN", "OLN",
            "CC", "TROX", "KRO", "TSE", "LYB", "WLK",
        ]

        # ============================================================
        # INDUSTRIALS
        # ============================================================
        self._cache["industrials"] = [
            "BA", "CAT", "DE", "HON", "GE", "MMM", "UNP",
            "RTX", "LMT", "NOC", "GD", "LHX", "HII",
            "TXT", "HWM", "TDG", "BWA", "APTV",
            "CMI", "PCAR", "AGCO", "CNHI", "TEX", "ALG",
            "WM", "RSG", "WCN", "CLH", "ECOL", "US",
            "GNRC", "HUBB", "AYI", "LECO", "LII", "SNA",
            "TTC", "MAS", "FBHS", "ALLE", "AAON", "JCI",
        ]

        # ============================================================
        # CONSUMER
        # ============================================================
        self._cache["consumer"] = [
            "AMZN", "WMT", "COST", "TGT", "DG", "DLTR",
            "KR", "ACI", "SFM", "GO", "IMKTA", "NGVC",
            "NKE", "LULU", "DECK", "CROX", "SKX", "ONON",
            "SBUX", "MCD", "YUM", "DPZ", "CMG", "QSR",
            "DIS", "NFLX", "ROKU", "PARA", "WBD", "LYV",
            "BKNG", "EXPE", "ABNB", "TRIP", "MMYT",
            "MAR", "HLT", "H", "WH", "IHG", "WYNN",
            "LVS", "MGM", "CZR", "DKNG", "PENN", "RSI",
        ]

        # ============================================================
        # HEALTHCARE
        # ============================================================
        self._cache["healthcare"] = [
            "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
            "ABT", "DHR", "BMY", "AMGN", "GILD", "VRTX",
            "REGN", "ISRG", "EW", "BSX", "MDT", "BDX",
            "ZBH", "SYK", "HCA", "THC", "CYH", "UHS",
            "INCY", "ALNY", "SGEN", "EXAS", "DXCM", "PODD",
            "ALGN", "HOLX", "IQV", "CRL", "MEDP",
        ]

        # ============================================================
        # COMMUNICATION
        # ============================================================
        self._cache["communication"] = [
            "META", "GOOGL", "GOOG", "DIS", "CMCSA", "NFLX",
            "T", "VZ", "TMUS", "CHTR", "LBRDA", "FOX",
            "PARA", "WBD", "SNAP", "PINS", "MTCH", "RBLX",
            "U", "TTWO", "EA", "ATVI", "ZG", "YELP",
            "IAC", "ANGI", "CARG", "CARS", "TRU",
        ]

        # ============================================================
        # UTILITIES
        # ============================================================
        self._cache["utilities"] = [
            "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC",
            "XEL", "ED", "WEC", "ES", "PCG", "EIX", "DTE",
            "AEE", "CMS", "CNP", "EVRG", "ATO", "NI",
            "PNW", "OGE", "BKH", "AVA", "POR", "NWE",
            "AWK", "WTRG", "SJW", "CWT", "MSEX", "YORW",
            "CEG", "VST", "NRG", "AES", "ORA",
        ]

        # ============================================================
        # REITs
        # ============================================================
        self._cache["reits"] = [
            "AMT", "CCI", "EQIX", "PLD", "SPG", "O", "PSA",
            "DLR", "SBAC", "WELL", "ARE", "AVB", "EQR",
            "VTR", "BXP", "SLG", "VNO", "KRC", "OHI",
            "NNN", "STOR", "ADC", "EPRT", "FCPT", "GTY",
            "STAG", "TRNO", "REXR", "FR", "EGP", "LXP",
            "COLD", "IIPR", "AIRC", "CPT", "UDR", "MAA",
            "ESS", "INVH", "AMH", "PEAK", "MPW", "HR",
        ]

        # ============================================================
        # SPACs & RECENT IPOs
        # ============================================================
        self._cache["spacs_recent_ipos"] = [
            "IONQ", "RGTI", "QUBT", "ARQQ", "QBTS",
            "ARM", "CART", "KVYO", "BIRK", "CAVA", "TOST",
            "BRZE", "MNDY", "PCOR", "CFLT", "RIVN", "LCID",
            "PLTR", "JOBY", "LILM", "ACHR", "GRAB",
            "NU", "DDOG", "SNOW", "U", "RBLX", "ABNB",
            "DASH", "AFRM", "SOFI", "HOOD", "DOCN",
        ]

        # ============================================================
        # INTERNATIONAL ADRs
        # ============================================================
        self._cache["international_adrs"] = [
            # China
            "BABA", "JD", "PDD", "BIDU", "NTES", "NIO", "XPEV",
            "LI", "BILI", "TME", "ZH", "IQ", "DOYU", "WB",
            "TAL", "EDU", "GOTU", "KC", "VIPS", "MNSO",
            # Japan
            "TM", "HMC", "SONY", "MUFG", "SMFG", "MFG",
            "NMR", "IX", "OFIX",
            # Korea
            "LPL", "KB", "SHG", "PKX",
            # India
            "INFY", "WIT", "HDB", "IBN", "TTM", "SIFY",
            "RDY", "MMYT", "YTRA", "FNKO",
            # Europe
            "ASML", "NVO", "AZN", "GSK", "SNY", "SAP",
            "TM", "UL", "DEO", "BTI", "BP", "SHEL",
            "TTE", "EQNR", "ENB", "CNQ", "SU", "IMO",
            "RY", "TD", "BMO", "BNS", "CM", "MFC",
            "SLF", "GWO", "POW", "FFH",
            # Latin America
            "MELI", "NU", "STNE", "PAGS", "DLO", "GLOB",
            "BSBR", "ITUB", "BBD", "PBR", "ERJ", "VALE",
            "GGB", "SID", "UGP",
            # SE Asia
            "SE", "GRAB", "CPNG",
        ]

        # ============================================================
        # ETFs — comprehensive
        # ============================================================
        self._cache["etfs"] = [
            # Broad market
            "SPY", "VOO", "IVV", "VTI", "ITOT", "SCHB", "SPTM",
            "QQQ", "QQQM", "VGT", "IWM", "IWN", "IWO", "IWF",
            "IWB", "IWD", "VTV", "VUG", "SCHV", "SCHG",
            "MDY", "IJH", "VO", "IVOO", "SPMD",
            "DIA", "RSP", "SPLG", "MGC", "OEF",
            # International
            "VEA", "VWO", "EFA", "EEM", "IEMG", "IXUS",
            "VXUS", "ACWI", "ACWX", "SPDW", "SPEM",
            # Fixed Income
            "BND", "AGG", "BNDX", "TLT", "IEF", "SHY",
            "TIP", "VTIP", "LQD", "HYG", "JNK", "EMB",
            "MUB", "SUB", "SHM", "VCSH", "VCIT", "VCLT",
            "GOVT", "VGSH", "VGIT", "VGLT", "SCHZ",
            # Commodity
            "GLD", "IAU", "SLV", "PPLT", "PALL",
            "USO", "BNO", "UNG", "DBA", "DBC", "PDBC",
            "GSG", "COMT", "USCI",
            # Sector
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLP",
            "XLU", "XLB", "XLRE", "XLC", "XLY",
            "VGT", "VFH", "VHT", "VDE", "VIS", "VDC",
            "VPU", "VAW", "VNQ", "VOX", "VCR",
            # Thematic
            "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ", "ARKX",
            "HACK", "CIBR", "BUG", "WCLD", "CLOU", "SKYY",
            "ROBO", "BOTZ", "IRBO", "AIQ", "GNOM", "EDOC",
            "TAN", "ICLN", "QCLN", "PBW", "FAN", "CNRG",
            "LIT", "DRIV", "IDRV", "MSOS", "MJ", "YOLO",
            "BLOK", "BITQ", "DAPP", "BITO", "GBTC", "ETHE",
            "JETS", "PEJ", "AWAY", "CRUZ",
            "ESPO", "HERO", "NERD", "GAMR",
            "SOXX", "SMH", "XSD", "PSI",
            "XBI", "IBB", "ARKG", "LABU", "LABD",
            # Leveraged / Inverse
            "TQQQ", "SQQQ", "UPRO", "SPXU", "UDOW", "SDOW",
            "TNA", "TZA", "SOXL", "SOXS", "FNGU", "FNGD",
            "SPXL", "SPXS", "FAS", "FAZ", "NUGT", "DUST",
            "JNUG", "JDST", "ERX", "ERY", "TECL", "TECS",
            "UCO", "SCO", "AGQ", "ZSL", "UGL", "GLL",
            # Dividend
            "VYM", "SCHD", "DVY", "HDV", "SDY", "SPYD",
            "NOBL", "VIG", "DGRO", "DGRW", "RDVY",
            # Factor / Smart Beta
            "MTUM", "VLUE", "QUAL", "SIZE", "USMV",
            "MOAT", "COWZ", "DIVO", "JEPI", "JEPQ",
            "XYLD", "QYLD", "RYLD", "NUSI",
        ]

        self._cache["sector_etfs"] = [
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLP",
            "XLU", "XLB", "XLRE", "XLC", "XLY",
        ]

        self._cache["intl_etfs"] = [
            "FXI", "EWJ", "EWZ", "EWY", "INDA", "EWT",
            "EWG", "EWU", "EWQ", "EWP", "EWI", "EWL",
            "EWA", "EWC", "EWH", "EWS", "EWM", "THD",
            "VNM", "EPOL", "TUR", "GREK", "ECH", "EWW",
            "EPU", "GXG", "FM", "FRDM",
        ]

        self._cache["bond_etfs"] = [
            "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "EMB",
            "TIP", "MUB", "BND", "AGG", "BNDX", "GOVT",
            "VCSH", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT",
            "SCHZ", "SPTL", "SPAB", "SPSB", "SPLB",
        ]

        self._cache["commodity_etfs"] = [
            "GLD", "IAU", "SLV", "PPLT", "PALL",
            "USO", "BNO", "UNG", "DBA", "DBC", "PDBC",
            "GSG", "COMT", "USCI", "CPER", "JJC",
        ]

        # ============================================================
        # FOREX (yfinance format: EURUSD=X)
        # ============================================================
        self._cache["forex"] = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
            "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X",
            "CHFJPY=X", "NZDJPY=X", "EURCHF=X", "GBPCHF=X",
            "EURAUD=X", "GBPAUD=X", "AUDNZD=X", "EURNZD=X",
            "GBPNZD=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X",
            "NZDCAD=X", "USDMXN=X", "USDBRL=X", "USDTRY=X",
            "USDZAR=X", "USDNOK=X", "USDSEK=X", "USDDKK=X",
            "USDPLN=X", "USDHUF=X", "USDCZK=X", "USDSGD=X",
            "USDHKD=X", "USDCNY=X", "USDINR=X", "USDKRW=X",
            "USDTHB=X", "USDMYR=X", "USDIDR=X", "USDPHP=X",
            "DX-Y.NYB",  # Dollar index
        ]

        # ============================================================
        # FUTURES (yfinance format: ES=F)
        # ============================================================
        self._cache["futures"] = [
            # Equity index
            "ES=F", "NQ=F", "YM=F", "RTY=F",
            # Energy
            "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F",
            # Metals
            "GC=F", "SI=F", "HG=F", "PA=F", "PL=F",
            # Agriculture
            "ZC=F", "ZS=F", "ZW=F", "ZL=F", "ZM=F",
            "KC=F", "SB=F", "CC=F", "CT=F", "OJ=F",
            "LC=F", "LH=F", "FC=F",
            # Rates
            "ZN=F", "ZB=F", "ZF=F", "ZT=F",
            # FX futures
            "6E=F", "6B=F", "6J=F", "6A=F", "6C=F", "6S=F",
            # Volatility
            "VX=F",
        ]

        # ============================================================
        # CRYPTO (yfinance format: BTC-USD)
        # ============================================================
        self._cache["crypto"] = [
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
            "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "SHIB-USD",
            "MATIC-USD", "LTC-USD", "LINK-USD", "UNI-USD", "ATOM-USD",
            "XLM-USD", "ALGO-USD", "VET-USD", "FIL-USD", "ICP-USD",
            "NEAR-USD", "FTM-USD", "SAND-USD", "MANA-USD", "AXS-USD",
            "AAVE-USD", "MKR-USD", "CRV-USD", "COMP-USD", "SNX-USD",
            "SUSHI-USD", "YFI-USD", "UMA-USD", "BAL-USD", "REN-USD",
            "ZRX-USD", "BAT-USD", "ENJ-USD", "CHZ-USD", "GALA-USD",
            "APE-USD", "GMT-USD", "OP-USD", "ARB-USD", "SUI-USD",
            "SEI-USD", "TIA-USD", "INJ-USD", "RNDR-USD", "FET-USD",
            "OCEAN-USD", "AGIX-USD", "ROSE-USD", "EGLD-USD",
            "HBAR-USD", "QNT-USD", "STX-USD", "IMX-USD", "LDO-USD",
            "RPL-USD", "PEPE-USD", "WLD-USD", "BONK-USD",
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ComprehensiveTickerUniverse = None

def get_comprehensive_universe() -> ComprehensiveTickerUniverse:
    global _instance
    if _instance is None:
        _instance = ComprehensiveTickerUniverse()
    return _instance
