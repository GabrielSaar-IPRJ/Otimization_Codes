import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x,y = sp.symbols('x y')
texto = input("Escreva a f(x,y), use x e y como a variavel, use '*' como simbolo de multiplicativo e '**' como potenciacao\nPara trigonometricas, use em ingles, como sin(x) = seno(x)\n")
func = sp.sympify(texto, locals={'e': sp.E})
func_numerico = sp.lambdify([x,y], func, 'numpy')
Array_X = []
Parametro_d = 1
Parametro_J = 2
Parametro_Beta = 0.5
limite = 1000
Contador_iter = 0
erro = float(input("Digite o erro desejado como criterio de parada: "))
pontos_iter = []
iter = []

for _ in range(3):
    X0 = float(input("Digite o valor de X: "))
    Y0 = float(input("DIgite o valor de Y: "))
    P0 = np.array([X0,Y0])
    Array_X.append(P0)
    print(f"Vetor {P0}, foi adicionado")

print("Xn = Pior array\nXs = Array Médio\nXl = Melhor Array")

def Organiza_Vetor(Array):
    melhor = 0
    medio = 0
    pior = 0
    for i in range(3):
        if(i == 0):
            melhor = Array[i]
            medio = Array[i]
            pior = Array[i]
        else:
            if(func_numerico(Array[i][0],Array[i][1]) < func_numerico(melhor[0],melhor[1])):
                medio = melhor
                melhor = Array[i]
            else:
                if(func_numerico(Array[i][0],Array[i][1]) >= func_numerico(pior[0],pior[1])):
                    medio = pior
                    pior = Array[i]
                else:
                    medio = Array[i]
    Array[0] = pior
    Array[1] = medio
    Array[2] = melhor
    print(f"Xn = {Array[0]}; Xs = {Array[1]}; Xl = {Array[2]}")
    return Array

def expandir(Xr,C):
    Xe = C + Parametro_J*(Xr-C)
    return Xe

def contrair(Xr,C):
    Xc = C + Parametro_Beta*(Xr-C)
    return Xc

def contrair_encolhido(Xl,Xj):
    Xm = Xl + Parametro_Beta*(Xj-Xl)
    return Xm

while(Contador_iter != limite):
    ##
    Array_X = Organiza_Vetor(Array_X)
    f_pior = func_numerico(Array_X[0][0],Array_X[0][1])
    f_medio = func_numerico(Array_X[1][0],Array_X[1][1])
    f_melhor = func_numerico(Array_X[2][0],Array_X[2][1])
    ##
    if(abs(f_melhor-f_pior) < erro):
        break
    centroide = (1/2)*(Array_X[1]+Array_X[2])
    Xr = centroide + Parametro_d*(centroide - Array_X[0])
    f_xr = func_numerico(Xr[0],Xr[1])
    ##
    if(f_xr <= f_medio and f_xr > f_melhor):
        Array_X[0] = Xr
    else:
        if(f_xr < f_melhor):
            Xe = expandir(Xr,centroide)
            f_xe = func_numerico(Xe[0],Xe[1])
            if(f_xe < f_xr):
                Array_X[0] = Xe
            else:
                Array_X[0] = Xr
        else:
            Xc = contrair(Xr,centroide)
            f_xc = func_numerico(Xc[0],Xc[1])
            if(f_xc < f_pior):
                Array_X[0] = Xc
            else:
                Array_X[0] = contrair_encolhido(Array_X[2],Array_X[0])
                Array_X[1] = contrair_encolhido(Array_X[2],Array_X[1])
    iter.append(Contador_iter)
    Contador_iter = Contador_iter + 1
    pontos_iter.append(f_melhor)

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel("Valor de (x,y)")
plt.xlabel("Iteração")
plt.title("Diferença de estimativa por iteração")
plt.plot(iter,pontos_iter)
plt.show()