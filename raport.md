# Zadanie 1
Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona. Algorytm Newtona powinien móc działać w dwóch trybach:
- ze stałym parametrem kroku
- z adaptacją parametru kroku przy użyciu metody z nawrotami.

Zbadaj wpływ wartości parametru kroku na zbieżność obu metod. Ponadto porównaj czasy działania obu algorytmów.

## Metoda gradientu prostego
1. Wpływ kroku na zbieżność\
Testowane wartości kroku:
```
step = np.linspace(0, 1, 8)
```
![zbierznosc_1](Figure_1.png)\
Możemy zaobserwować, że optymalna wielkość kroku mieści się między 0.28 - 0.43.\
![zbierznosc](Figure_2.png)\
Najlepsze działanie dla kroku między 0.7 i 0.87.\
![zbierznosc](Figure_3.png)\
Wyniki dla większości wartości kroku są porównywalne, algorytm działa najlepiej dla wartości z przedziału [0.28, 0.57].\
![zbierznosc](Figure_4.png)\
Widzimy, że im mniejszy krok, tym lepsze wyniki, więc zbadajmy działanie dla przedziału [0, 0.1]:\
![zbierznosc](Figure_5.png)\
Najlepsza zbieżność dla kroku z przedziału [0.05, 0.07]\
![zbierznosc](Figure_6.png)
![zbierznosc](Figure_7.png)



