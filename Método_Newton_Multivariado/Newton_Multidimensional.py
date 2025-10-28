import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')
texto = input(
    "Escreva a f(x,y), use x e y como a variavel, use '*' como simbolo de multiplicativo e '**' como potenciacao\n"
    "Para trigonometricas, use em ingles, como sin(x) = seno(x)\n"
)
func_analitic = sp.sympify(texto, locals={'e': sp.E})
func = sp.lambdify((x, y), func_analitic, 'numpy')

##
gradriente_sym= sp.Matrix([sp.diff(func_analitic, x), sp.diff(func_analitic, y)])
Matriz_sym = sp.Matrix([
    [sp.diff(func_analitic, x, 2), sp.diff(func_analitic, x, y)],
    [sp.diff(func_analitic, y, x), sp.diff(func_analitic, y, 2)]
])

##
gradiente_num= sp.lambdify((x, y), gradriente_sym, 'numpy')
Matriz_num = sp.lambdify((x, y), Matriz_sym, 'numpy')
##

x_inicial = float(input("Digite o valor de X inicial: "))
y_inicial = float(input("Digite o valor de Y inicial: "))
ponto_atual = np.array([x_inicial, y_inicial], dtype=float)

erro_max = 10**(-9)
max_iter = 200
i = 0
erros = []
iteracoes = []

while i < max_iter:
    gradiente_pontual = np.array(gradiente_num(ponto_atual[0], ponto_atual[1]), dtype=float).flatten()
    matriz_pontual = np.array(Matriz_num(ponto_atual[0], ponto_atual[1]), dtype=float)
    try:   
        matriz_inv = np.linalg.inv(matriz_pontual)
    except np.linalg.LinAlgError:
        matriz_inv = np.linalg.pinv(matriz_pontual)
    ponto_futuro = ponto_atual - np.dot(matriz_inv, gradiente_pontual)
    erro = np.linalg.norm(ponto_futuro - ponto_atual)
    erros.append(erro)
    iteracoes.append(i)
    print(f"Iteração {i}: x={ponto_futuro[0]:.6f}, y={ponto_futuro[1]:.6f}, erro={erro:.2e}")
    if erro < erro_max:
        print(f"Ponto mínimo encontrado em x={ponto_futuro[0]:.5f}, y={ponto_futuro[1]:.5f}")
        break
    ponto_atual = ponto_futuro
    i += 1

# Plot do erro
if erros:
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.semilogy(iteracoes, erros)
    plt.xlabel('Iteração')
    plt.ylabel('Erro')
    plt.title('Convergência do Método de Newton')
    plt.show()