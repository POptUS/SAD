import numpy as np
import pycutest

#########################################
# SET YOUR QUERY PARAMETERS HERE
#########################################

filters = {"constraints": "unconstrained", "regular": True, "n": [2, 15], "userN": True}  # Unconstrained problems

probstrings = pycutest.find_problems(**filters)

nprobs = len(probstrings)

#########################################
# END QUERY
#########################################

from pycutest import import_problem


def make_pyCUTEst_problem_list():

    # Create problem instances
    engval1 = import_problem("ENGVAL1", sifParams={"N": 2})
    powellsg = import_problem("POWELLSG", sifParams={"N": 4})
    trigon1 = import_problem("TRIGON1", sifParams={"N": 10})
    arglinc = import_problem("ARGLINC", sifParams={"N": 10, "M": 20})  # changed from M=20
    inteqnels = import_problem("INTEQNELS", sifParams={"N": 10})
    sensors = import_problem("SENSORS", sifParams={"N": 2})
    eigenbls = import_problem("EIGENBLS", sifParams={"N": 2})
    indefm = import_problem("INDEFM", sifParams={"N": 10})  # letting ALPHA be default 0.5
    dqdrtic = import_problem("DQDRTIC", sifParams={"N": 10})
    fletcbv3 = import_problem("FLETCBV3", sifParams={"N": 10, "KAPPA": 0})
    modbeale = import_problem("MODBEALE", sifParams={"N/2": 1, "ALPHA": 50})
    broyndbdls = import_problem("BROYDNBDLS", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 5, "KAPPA3": 1, "LB": 5, "UB": 1})
    sparsqur = import_problem("SPARSQUR", sifParams={"N": 10})
    hilberta = import_problem("HILBERTA", sifParams={"N": 2, "D": 0})
    dixmaanf = import_problem("DIXMAANF", sifParams={"M": 5})  # N=15
    chainwoo = import_problem("CHAINWOO", sifParams={"NS": 1})
    dixmaanh = import_problem("DIXMAANH", sifParams={"M": 5})  # N=15
    errinros = import_problem("ERRINROS", sifParams={"N": 10})
    strtchdv = import_problem("STRTCHDV", sifParams={"N": 10})
    mancino = import_problem("MANCINO", sifParams={"N": 10, "ALPHA": 5, "BETA": 14, "GAMMA": 3})
    dixon3dq = import_problem("DIXON3DQ", sifParams={"N": 10})
    dixmaano = import_problem("DIXMAANO", sifParams={"M": 5})  # N=15

    chnrosnb = import_problem("CHNROSNB", sifParams={"N": 10})
    woods = import_problem("WOODS", sifParams={"NS": 1})
    sinquad2 = import_problem("SINQUAD2", sifParams={"N": 5})
    arglinb = import_problem("ARGLINB", sifParams={"N": 10, "M": 20})
    morebv = import_problem("MOREBV", sifParams={"N": 10})
    penalty2 = import_problem("PENALTY2", sifParams={"N": 4})
    fletchcr = import_problem("FLETCHCR", sifParams={"N": 10})
    sparsine = import_problem("SPARSINE", sifParams={"N": 10})
    dixmaann = import_problem("DIXMAANN", sifParams={"M": 5})  # N=15
    sbrybnd = import_problem("SBRYBND", sifParams={"N": 10})
    dixmaang = import_problem("DIXMAANG", sifParams={"M": 5})  # N=15
    noncvxu2 = import_problem("NONCVXU2", sifParams={"N": 10})
    fletcbv2 = import_problem("FLETCBV2", sifParams={"N": 10, "KAPPA": 1})
    eigenals = import_problem("EIGENALS", sifParams={"N": 2})
    broydn7d = import_problem("BROYDN7D", sifParams={"N/2": 5})
    dixmaanj = import_problem("DIXMAANJ", sifParams={"M": 5})  # N=15

    broydn3dls = import_problem("BROYDN3DLS", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 1})
    #    scurly20 = import_problem('SCURLY20', sifParams={'N': 10}) # causes error???
    dixmaanm1 = import_problem("DIXMAANM1", sifParams={"M": 5})  # N=15
    fletbv3m = import_problem("FLETBV3M", sifParams={"N": 10, "KAPPA": 1})
    kssls = import_problem("KSSLS", sifParams={"N": 4})
    box = import_problem("BOX", sifParams={"N": 10})
    #    luksan17ls = import_problem('LUKSAN17LS')  # This problem doesn't have adjustable parameters
    dixmaanc = import_problem("DIXMAANC", sifParams={"M": 5})  # N=15
    dixmaand = import_problem("DIXMAAND", sifParams={"M": 5})  # N=15
    msqrtals = import_problem("MSQRTALS", sifParams={"P": 2})  # N=4
    powersum = import_problem("POWERSUM", sifParams={"N": 4})
    testquad = import_problem("TESTQUAD", sifParams={"N": 10})
    dixmaani1 = import_problem("DIXMAANI1", sifParams={"M": 5})  # N=15
    #    spinls = import_problem('SPINLS', sifParams={'N': 2})# matrix dimension? # gives an Inf

    brybnd = import_problem("BRYBND", sifParams={"N": 10, "KAPPA1": 2, "KAPPA2": 5, "KAPPA3": 1, "LB": 5, "UB": 1})
    oscipath = import_problem("OSCIPATH", sifParams={"N": 2, "RHO": 1})
    vardim = import_problem("VARDIM", sifParams={"N": 10})
    tointgss = import_problem("TOINTGSS", sifParams={"N": 10})
    cosine = import_problem("COSINE", sifParams={"N": 10})
    spin2ls = import_problem("SPIN2LS", sifParams={"N": 2})  # matrix dimension?
    dixmaane1 = import_problem("DIXMAANE1", sifParams={"M": 5})  # N=15
    arglina = import_problem("ARGLINA", sifParams={"N": 10, "M": 20})
    ssbrybnd = import_problem("SSBRYBND", sifParams={"N": 10})
    noncvxun = import_problem("NONCVXUN", sifParams={"N": 10})
    argtrigls = import_problem("ARGTRIGLS", sifParams={"N": 10})
    dixmaana1 = import_problem("DIXMAANA1", sifParams={"M": 5})  # N=15
    penalty1 = import_problem("PENALTY1", sifParams={"N": 4})
    yatp2cls = import_problem("YATP2CLS", sifParams={"N": 2})  # n=8
    yatp2ls = import_problem("YATP2LS", sifParams={"N": 2})  # n=8
    dixmaanp = import_problem("DIXMAANP", sifParams={"M": 5})  # N=15
    power = import_problem("POWER", sifParams={"N": 10})
    sinquad = import_problem("SINQUAD", sifParams={"N": 5})
    srosenbr = import_problem("SROSENBR", sifParams={"N/2": 5})
    # indef = import_problem("INDEF", sifParams={"N": 10})  # using default ALPHA=0.5 # not bounded below
    tquartic = import_problem("TQUARTIC", sifParams={"N": 5})
    dqrtic = import_problem("DQRTIC", sifParams={"N": 10})
    schmvet = import_problem("SCHMVETT", sifParams={"N": 3})
    extrosnb = import_problem("EXTROSNB", sifParams={"N": 5})
    hilbertb = import_problem("HILBERTB", sifParams={"N": 5, "D": 5})
    dixmaanb = import_problem("DIXMAANB", sifParams={"M": 5})  # N=15

    boxpower = import_problem("BOXPOWER", sifParams={"N": 10})
    dixmaanl = import_problem("DIXMAANL", sifParams={"M": 5})  # N=15
    nondia = import_problem("NONDIA", sifParams={"N": 10})
    dixmaank = import_problem("DIXMAANK", sifParams={"M": 5})  # N=15
    tridia = import_problem("TRIDIA", sifParams={"N": 10})
    watson = import_problem("WATSON", sifParams={"N": 12})
    trigon2 = import_problem("TRIGON2", sifParams={"N": 10})
    #    scurly30 = import_problem('SCURLY30', sifParams={'N': 10})# causes error???
    freuroth = import_problem("FREUROTH", sifParams={"N": 2})
    genhumps = import_problem("GENHUMPS", sifParams={"N": 5})
    genrose = import_problem("GENROSE", sifParams={"N": 5})
    cragglvy = import_problem("CRAGGLVY", sifParams={"M": 1})  # N=4
    chnrsnbm = import_problem("CHNRSNBM", sifParams={"N": 10})
    msqrtbls = import_problem("MSQRTBLS", sifParams={"P": 3})  # N=9
    errinrsm = import_problem("ERRINRSM", sifParams={"N": 10})
    brownal = import_problem("BROWNAL", sifParams={"N": 10})

    #########################################################################
    #########################################################################
    #########################################################################

    # Create list of problem instances
    problems = [
        engval1,
        powellsg,
        trigon1,
        arglinc,
        inteqnels,
        sensors,
        eigenbls,
        indefm,
        dqdrtic,
        fletcbv3,
        modbeale,
        broyndbdls,
        sparsqur,
        hilberta,
        dixmaanf,
        chainwoo,
        dixmaanh,
        errinros,
        strtchdv,
        mancino,
        dixon3dq,
        dixmaano,
        chnrosnb,
        woods,
        sinquad2,
        arglinb,
        morebv,
        penalty2,
        fletchcr,
        sparsine,
        dixmaann,
        sbrybnd,
        dixmaang,
        noncvxu2,
        fletcbv2,
        eigenals,
        broydn7d,
        dixmaanj,
        broydn3dls,
        #        scurly20,  # causes error???
        dixmaanm1,
        fletbv3m,
        kssls,
        box,
        #        luksan17ls,  # This problem doesn't have adjustable parameters
        dixmaanc,
        dixmaand,
        msqrtals,
        powersum,
        testquad,
        dixmaani1,
        #        spinls,  # matrix dimension? # gives an Inf
        brybnd,
        oscipath,
        vardim,
        tointgss,
        cosine,
        spin2ls,
        dixmaane1,
        arglina,
        ssbrybnd,
        noncvxun,
        argtrigls,
        dixmaana1,
        penalty1,
        yatp2cls,
        yatp2ls,
        dixmaanp,
        power,
        sinquad,
        srosenbr,
        # indef,  # using default ALPHA=0.5 # not bounded below
        tquartic,
        dqrtic,
        schmvet,
        extrosnb,
        hilbertb,
        dixmaanb,
        boxpower,
        dixmaanl,
        nondia,
        dixmaank,
        tridia,
        watson,
        trigon2,
        #        scurly30,  # causes error???
        freuroth,
        genhumps,
        genrose,
        cragglvy,
        chnrsnbm,
        msqrtbls,
        errinrsm,
        brownal,
    ]

    print(f"{len(problems)} CUTEst problems have been loaded.")

    return problems
