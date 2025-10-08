import matplotlib.pyplot as plt
import sympy



texto = input("Escreva a f(x), use x como a variavel, use '*' como simbolo de multiplicativo e '**' como potenciacao\n Para trigonometricas use os termos em ingles, como sin para seno\n")
func = sympy.sympify(texto, locals={'e': sympy.E})
X = sympy.Symbol("x")
xi = float(input("Digite o valor de x inicial: \n"))
c_parada = float(input("Digite a quantidade de casas decimais de erro desejado\n"))
h= 10**(-9)
func_derv = sympy.diff(func,X)
func_derv_segunda = sympy.diff(func,X,2)
erros = []
iteracao = []
i = 0
print("|Iter|Xi|Xii|f(x1)|erro|")

while True:
    if(func_derv_segunda.subs(X,xi) == 0):
        print("Segunda derivada igual a zero\n")
        txt = "X estimativo de: {:.5f}"
        print(txt.format(xi))  
        break
    else:      
        xi_proximo = xi - (func_derv.subs(X,xi)/func_derv_segunda.subs(X,xi))
        erro = abs(xi_proximo-xi)
        erros.append(erro)
        i = i+1
        iteracao.append(i)
        txt = "|{}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|"
        valor_func = float(func.subs(X,xi_proximo))
        print(txt.format(i,xi,xi_proximo,valor_func,erro))
        xi = xi_proximo
        if(erro < (10**(-c_parada))):
            txt = "X estimativo de: {:.5f}"
            print(txt.format(xi))
            break

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel("Diferença entre x1 e x2")
plt.xlabel("Iteração")
plt.title("Diferença de estimativa por iteração")
plt.plot(iteracao,erros)
plt.show()