import numpy as np

from symmetric_matrix2vec import symmetric_matrix2vec
from multiindex2derivatives import multiindex2derivatives


def call_fun_script(fun, x, alpha_admissible, n, fhist, gradhist, Hesshist, flag, nfs):
    """
    evaluate the user-supplied function fun
    assert that the API of fun obeys SAD formatting conventions
    coerce data structures to the correct format when possible

    INPUTS:
    fun      [f h] function to evaluate
    x        [dbl] [n] point at which to evaluate fun
    alpha_admissible [int] [list] multi-indices of available partial derivatives
    n        [int] dimension of the input variable, x
    fhist    [dbl] [nfmax] previous history of evaluations of fun
    gradhist [dbl] [nfmax-by-n] previous history of evaluations of the gradient of fun
    Hesshist [dbl] [nfmax-by-n-by-n] previous history of evaluations of the Hessian of fun
    nfs      [int] number of evaluations of fun prior to this script
                   AKA -- the number of nontrivial entries in fhist
    ndfs     [int] number of evaluations of the gradient and Hessian of fun
                   prior to this script
                   AKA -- the number of nontrivial rows in gradhist
                   and the number of nontrivial axis 0 entries in Hesshist
    only_fval [bool] True -> only query fun for the function value
                    False -> also query for the gradient and Hessian
    only_gradHess [bool] True -> only query fun for the gradient and Hessian
                         False -> also query for the function value

    OUTPUTS:
    fhist    [dbl] function value history, appended with new query
    gradhist [dbl] gradient ''''''''''''''''''''''''''''''''''''''
    Hesshist [dbl] Hessian '''''''''''''''''''''''''''''''''''''''
        If no gradient or no Hessian components are available, the corresponding
        histories are left unchanged.
    flag
    nfs      [int] number of function evaluations after this script
    """

    # process available derivatives
    assert type(alpha_admissible) is list
    assert (0,)*n in alpha_admissible # must have function value available
    degrees = [sum(alpha) for alpha in alpha_admissible]
    
    # number of derivatives available for each derivative degree
    n0 = degrees.count(0)
    n1 = degrees.count(1)
    n2 = degrees.count(2)
    assert n0 == 1
    assert n1 <= n
    assert n2 <= n * (n +1) / 2

    # coerce format of x
    assert isinstance(x, np.ndarray)
    assert issubclass(x.dtype.type, np.floating)
    x = np.atleast_1d(x)
    if n > 1:
        x = np.squeeze(x)
    assert x.ndim == 1

    # call the function
    fnew, gradnew, Hessnew = fun(x)

    ########################################
    # look at output formats
    # and only save partial derivatives in K
    ########################################

    #############################
    # function value
    #############################

    # if function value is returned as a singleton array
    if isinstance(fnew, np.ndarray):
        fnew = np.squeeze(fnew)
    assert isinstance(fnew, float)

    #############################
    # gradient
    #############################

    if n1 > 0:

        assert isinstance(gradnew, np.ndarray)
        assert issubclass(gradnew.dtype.type, np.floating)
        gradnew = np.atleast_1d(gradnew)
        gradnew = np.squeeze(gradnew)
        assert gradnew.ndim == 1

        # process the shapes of gradient and Hessian output in terms of K
        if len(gradnew) == n: # user supplied full gradient; trim 
            K = []
            for i in range(len(alpha_admissible)):
                if degrees[i] == 1:
                    alpha = alpha_admissible[i]
                    K.append( multiindex2derivatives(alpha)[0] )

            gradnew = gradnew[K]
        else:
            assert gradnew.shape == (n1,)

    #############################
    # Hessian
    #############################

    if n2 > 0:

        assert isinstance(Hessnew, np.ndarray)
        assert issubclass(Hessnew.dtype.type, np.floating)
        
        # allow for matrix output or vectorized upper triangular output
        if Hessnew.ndim == 1:

            if len(Hessnew) == n * (n + 1) // 2:  # full upper triangular entries of Hessian instead of just known components block
                
                Hessnewmat = vec2symmetric_matrix(Hessnew, n)

                temp = np.zeros(n2)
                count = 0
                for i in range(len(alpha_admissible)):
                    if degrees[i] == 2:
                        alpha = alpha_admissible[i]
                        k1, k2 = multiindex2derivatives(alpha)
                        temp[count] = Hessnewmat[k1, k2]
                        count += 1

                Hessnew = temp

            else:
                assert len(Hessnew) == n2  # already in desired format

        else: # Hessian inputted as a matrix

            Hessnew = np.atleast_2d(Hessnew)
            Hessnew = np.squeeze(Hessnew)
            assert Hessnew.ndim == 2

            if Hessnew.shape == (n, n):  # full symmetric Hessian as matrix

                temp = np.zeros(n2)
                count = 0
                for i in range(len(alpha_admissible)):
                    if degrees[i] == 2:
                        alpha = alpha_admissible[i]
                        k1, k2 = multiindex2derivatives(alpha)
                        temp[count] = Hessnew[k1, k2]
                        count += 1

                Hessnew = temp

            else:
                print("Unknown Hessian format encountered.")

    ####################################
    # insert into histories
    ####################################

    assert fhist[nfs] == 0
    fhist[nfs] = fnew
    if n1 > 0:
        assert all(gradhist[nfs] == np.zeros(n1))
        gradhist[nfs] = gradnew
    if n2 > 0:
        assert all(Hesshist[nfs] == np.zeros(n2))
        Hesshist[nfs] = Hessnew
    
    nfs += 1

    # handle function values of NaN or Inf
    if not(np.isfinite(fnew)):
        flag = 4
    if n1>0:
        if not(np.isfinite(gradnew).all()):
            flag = 4
    if n2>0:
        if not(np.isfinite(Hessnew).all()):
            flag = 4



    return fhist, gradhist, Hesshist, flag, nfs
