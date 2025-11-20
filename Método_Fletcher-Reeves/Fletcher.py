import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def Beta_calc(xf,xi):
    gradiente_novo = gradiente_num(xf[0],xf[1])
    gradiente_anterior = gradiente_num(xi[0],xi[1])
    numerador = np.dot(np.transpose(gradiente_novo),gradiente_novo)
    denominador = np.dot(np.transpose(gradiente_anterior),gradiente_anterior)
    return numerador/denominador

def Newton_Raphson(func_t):
    dfunc_simbolica = sp.diff(func_t,t)
    d2func_simbolica = sp.diff(func_t,t,2)
    dfunc = sp.lambdify(t, dfunc_simbolica, 'numpy')
    d2func = sp.lambdify(t, d2func_simbolica, 'numpy')
    ti = 0
    erro = 10**(-9)
    j = 0
    while(j < 150):
        d1 = float(dfunc(ti))
        d2 = float(d2func(ti))

        # proteger d2 muito pequeno
        if abs(d2) < 10**(-9):
            d2_adj = 10**(-9)
        else:
            d2_adj = d2
        passo = d1 / d2_adj
        if not np.isfinite(passo):
            return ti
        if abs(passo) > 2:
            passo = np.sign(passo) * 2
        tii = ti - passo
        if abs(tii - ti) < erro:
            return tii
        ti = tii
        j += 1
    return ti


x,y,t = sp.symbols('x y t')
texto = input("Escreva a f(x,y), use x e y como a variavel, use '*' como simbolo de multiplicativo e '**' como potenciacao\nPara trigonometricas, use em ingles, como sin(x) = seno(x)\n")
##
func = sp.sympify(texto, locals={'e': sp.E})
func_numerico = sp.lambdify([x,y], func, 'numpy')
##
gradiente= [sp.diff(func,x),sp.diff(func,y)]
gradiente_num = sp.lambdify((x,y), gradiente, 'numpy')
##
x_inicial = float(input("Digite o valor de X incial: "))
y_incial= float(input("Digite o valor de Y inicia: "))
ponto_atual=np.array([x_inicial,y_incial], dtype=float)
##
grad =np.array(gradiente_num(ponto_atual[0],ponto_atual[1]))
dk = (-1)*grad
##
erro_min= 10**(-5)
i = 0
erros = []
iteracoes = []
max_iter = 2000

while i <= max_iter:
    ponto_minimizar = ponto_atual + (t*dk)
    func_t = func.subs([(x,ponto_minimizar[0]), (y,ponto_minimizar[1])])
    lambd = Newton_Raphson(func_t)
    ponto = ponto_atual + (lambd*dk)
    Bk = Beta_calc(ponto,ponto_atual)
    grad =np.array(gradiente_num(ponto[0],ponto[1]))
    dk = (-1)*grad + Bk*dk
    erro = np.linalg.norm(ponto - ponto_atual)
    erros.append(erro)
    iteracoes.append(i)
    if(erro < erro_min):
        text = "Após {} iterações, os valores de minimo são\nX={:.5f}\nY={:.5f}".format(i,ponto[0],ponto[1])
        print(text)
        break
    ponto_atual= ponto
    text = "Iteração {}, x={:.5f}, y={:.5f}".format(i,ponto[0],ponto[1])
    i = i + 1
    print(text)        
iteracoes_inteiras = [int(i) for i in iteracoes]
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel("Diferença entre f(xii,yii) e f(xi,yi)")
plt.xlabel("Iteração")
plt.title("Diferença de estimativa por iteração")
plt.plot(iteracoes_inteiras,erros)
plt.show()