import copy

class Standardization:
    def __init__(self, X, T = None):
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self._setup_standardize(X, T)
        
    def _setup_standardize(self, X, T = None):
        self.Xmeans = X.mean(axis = 0)
        self.Xstds = X.std(axis = 0)
        self.Xconstant = self.Xstds == 0
        self.XstdsFixed = copy.copy(self.Xstds)
        self.XstdsFixed[self.Xconstant] = 1
        
        if T is not None:
            self.Tmeans = T.mean(axis = 0)
            self.Tstds = T.std(axis = 0)

    def _standardizeX(self, X):
        Xst = (X - self.Xmeans) / self.XstdsFixed
        Xst[self.Xconstant] = 0.0
        return Xst

    def _standardizeT(self, T):
        Tst = (T - self.Tmeans) / self.Tstds
        return Tst

    def _unstandardizeX(self, Xst):
        X = (Xst * self.Xstds) + self.Xmeans
        return X

    def _unstandardizeT(self, Tst):
        T = (Tst * self.Tstds) + self.Tmeans
        return T
