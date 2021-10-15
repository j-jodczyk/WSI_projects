# Zadanie 1. (7p)
# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona.
# Algorytm Newtona powinien móc działać w dwóch trybach:
#       ze stałym parametrem kroku
#       z adaptacją parametru kroku przy użyciu metody z nawrotami.
#           w iteracyjny sposób znaleźć


# x to chyba powinien byc wektor, bo funkcja wieloargumentowa
def gradientDescend(x0, grad, step, error_margin=0.0001):
    # krok stały, podany przez użytkownika
    # x nalezy do D = [-100, 100]^n
    x = x0
    diff = x0
    while diff>error_margin:
        prev_x = x
        x = prev_x - step*grad(prev_x)
        diff = abs(prev_x-x)


# x wektor
def newtonConstStep(x0, grad, inv_hess, step, error_margin=0.0001):
    x = x0
    diff = x0
    prev_x = x
    while diff>error_margin:
        prev_x = x
        x = prev_x - step*inv_hess(prev_x)*grad(x)
        diff = abs(prev_x-x)


# x wektor
def newtonAdaptStep(x0, func, grad, inv_hess, step, error_margin=0.0001):
    x = x0
    diff = x0
    prev_x = x
    while func(x+t*v)>func(x)+alpha*t*grad(x)^T*v:
        t = betha*t
    while diff>error_margin:
        prev_x = x
        x = prev_x - t*step*inv_hess(prev_x)*grad(x)
        diff = abs(prev_x-x)


def main():
    pass


if __name__=="__main__":
    main()

