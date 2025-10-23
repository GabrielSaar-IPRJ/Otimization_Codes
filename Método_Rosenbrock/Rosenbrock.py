import numpy as np
import sympy
import matplotlib.pyplot as plt
import math

x,y,t = sympy.symbols('x y t')
texto = input("Escreva a f(x,y), use x e y como a variavel, use '*' como simbolo de multiplicativo e '**' como potenciacao\nPara trigonometricas, use em ingles, como sin(x) = seno(x)\n")
func = sympy.sympify(texto, locals={'e': sympy.E})
func_numerico = sympy.lambdify([x,y], func, 'numpy')
d1 = np.array([1,0])
d2 = np.array([0,1])
dj = np.array([d1,d2])
erros = []
iteracoes = []
i = 1
x0 = float(input("Digite o valor de X0\n"))
y0 = float(input("Digite o valor de Y0\n"))
yi = np.array([x0,y0])
erro = float(input("Digite o erro minimo de parada\n"))
max_iter = 1000

##Funções para o método
def Newton_Raphson(func_t,max_iter):
    dfunc_simbolica = sympy.diff(func_t,t)
    d2func_simbolica = sympy.diff(func_t,t,2)
    dfunc = sympy.lambdify(t, dfunc_simbolica, 'numpy')
    d2func = sympy.lambdify(t, d2func_simbolica, 'numpy')
    ti = 0
    erro = 10**(-9)
    j = 0
    while(j < max_iter):
        if(d2func(ti) != 0):
            tii = ti - (dfunc(ti)/d2func(ti))
        else:
            return ti
        if(abs(tii-ti) < erro):
           return tii
        ti = tii
        j = j + 1
    return ti

def Aj (lambdj,dj):
    j = 0
    aj = []
    n = 1 
    while j <= n:
        if(lambdj[j] == 0):
            aj.append(dj[j])
        else:
            i = j
            soma = np.array([0,0])
            for i in range (n):
                soma = soma+(lambdj[i]*dj[i])
            aj.append(soma)
        j = j + 1
    return aj

def dj_barra (aj):
    j = 0
    n = 1
    bj = []
    dj_barras = []
    while j <= n:
        if j == 0:
            bj.append(aj[j])
            norma = math.sqrt((bj[j][0])**2 + (bj[j][1])**2)
            if norma != 0:
                d_barra = bj[j]/norma
            else:
                d_barra = bj[j]
            dj_barras.append(d_barra)
        else:
            d_barra = bj[j-1]/math.sqrt((bj[j-1][0])**2 + (bj[j-1][1])**2)
            soma = aj[j] - (aj[j])*(d_barra.T)*(d_barra)
            bj.append(soma)
            norma = math.sqrt((bj[j][0])**2 + (bj[j][1])**2)
            if norma != 0:
                d_barra = bj[j]/math.sqrt((bj[j][0])**2 + (bj[j][1])**2)
            else:
                d_barra = bj[j]
            dj_barras.append(d_barra)
        j = j+1
    return dj_barras

##inicio do processo iterativo
while (i != max_iter):
    ##Primeira busca
    p1 = yi+(t * dj[0])
    func_t = func.subs({x:p1[0], y:p1[1]})
    lambd1 = Newton_Raphson(func_t,max_iter)
    p1 = yi+(lambd1 * dj[0])
    ##Segunda busca
    p2 = p1+(t*dj[1])
    func_t = func.subs({x:p2[0], y:p2[1]})
    lambd2 = Newton_Raphson(func_t,max_iter)
    p2 = p1+(lambd2 * dj[1])
    ##Definição dos vetores de avanço
    lamdj = np.array([lambd1,lambd2])
    aj = Aj(lamdj,dj)
    erro_iter = abs(func_numerico(p2[0],p2[1])-func_numerico(yi[0],yi[1]))
    erros.append(erro_iter)
    iteracoes.append(i)
    if(lambd1 < 10**(-9) and lambd2 < 10**(-9)):
        print("Convergencia atinjida por passo nulo, possivel minimo local encontrado")
        texto = 'valor de x e y minimos\nX:{:.5f}\nY:{:.5f}\nf(x,y)={:.5f}'
        print(texto.format(p2[0],p2[1],func_numerico(p2[0],p2[1])))
        break
    distancias_novas = dj_barra(aj)
    dj[0] = distancias_novas[0]
    dj[1] = distancias_novas[1]
    if not np.isfinite(erro_iter):
        print("Erro numérico detectado, interrompendo.\n")
        texto = 'valor de x e y minimos\nX:{:.5f}\nY:{:.5f}\nf(x,y)={:.5f}'
        print(texto.format(p2[0],p2[1],func_numerico(p2[0],p2[1])))
        break
    if erro_iter < erro:
        texto = 'valor de x e y minimos\nX:{:.5f}\nY:{:.5f}\nf(x,y)={:.5f}'
        print(texto.format(p2[0],p2[1],func_numerico(p2[0],p2[1])))
        break
    i = i + 1
    yi = p2

iteracoes_inteiras = [int(i) for i in iteracoes]
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel("Diferença entre f(xii,yii) e f(xi,yi)")
plt.xlabel("Iteração")
plt.title("Diferença de estimativa por iteração")
plt.plot(iteracoes_inteiras,erros)
plt.show()