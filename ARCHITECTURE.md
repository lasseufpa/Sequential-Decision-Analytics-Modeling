# Arquitetura do `pysdm`

> **Parte I — Arquitetura implementada (v1)**: documenta o pacote que existe em
> `src/pysdm/` (código real, 31 testes passando, exemplos executáveis). É a referência
> para quem vai usar ou contribuir com a biblioteca.
>
> **Parte II — Log de design (histórico)**: a proposta pré-implementação e as decisões que
> a antecederam, preservada como registro de *por que* a arquitetura é o que é. As
> referências cruzadas internas ("seção 4.1", etc.) apontam para dentro da Parte II.
>
> Comentários do professor e o mapeamento 1:1 comentário → implementação estão em
> `PROPOSAL.md`, seção 10.

---

# Parte I — Arquitetura implementada (v1)

## I.1. Estrutura do projeto

Layout `src/` (o pacote não fica na raiz do repositório): evita importar acidentalmente a
árvore de fontes não instalada durante os testes — o teste roda contra o pacote instalado,
como o usuário final o veria. Padrão recomendado por packaging.python.org e usado por
`attrs`/`hypothesis`/`black`.

```
Sequential-Decision-Analytics-Modeling/
├── pyproject.toml              # metadados, dependências, config de ferramentas (PEP 621)
├── ARCHITECTURE.md             # este documento
├── PROPOSAL.md                 # proposta condensada + mapa comentário→implementação
│
├── src/
│   └── pysdm/
│       ├── __init__.py         # API pública: re-exporta tudo e define __all__/__version__
│       │
│       │   # ---- núcleo: o contrato matemático (estável, sem dependência pesada) ----
│       ├── elements.py         # State, Decision, ExogenousInfo (bases declarativas)
│       ├── model.py            # Model (ABC) — os 5 elementos do UMF + validação de assinatura
│       ├── exogenous.py        # ExogenousSource + ModelSource/DatasetSource/CallableSource
│       ├── exceptions.py       # hierarquia de exceções (PysdmError na raiz)
│       ├── _random.py          # check_random_state / spawn_generators (privado)
│       │
│       │   # ---- estratégias de decisão ----
│       ├── policies/
│       │   ├── __init__.py     # re-exporta as políticas concretas
│       │   ├── base.py         # Policy (ABC) — contrato único (decide + hooks + params)
│       │   ├── known.py        # Threshold, Greedy, UCB, IE, ThompsonSampling
│       │   └── adp.py          # ForwardADPPolicy + TabularValueFunction (seção 8.6.4)
│       │
│       │   # ---- execução, medição, observação ----
│       ├── engine.py           # Engine — o "motor": prepara tudo e roda o laço (template method)
│       ├── result.py           # RunResult — objeto rico devolvido por engine.run()
│       ├── history.py          # StepRecord + History (registro passo-a-passo, to_frame())
│       ├── metrics.py          # Metric (ABC) + MetricSet + métricas built-in
│       └── callbacks.py        # Callback (ABC) + ProgressLogger (observação online)
│
├── tests/                      # pytest; espelha a estrutura de src/ por assunto
│   ├── conftest.py             # fixtures compartilhadas (InventoryModel, ThresholdPolicy)
│   ├── test_elements.py        # State/Decision/ExogenousInfo declarativos
│   ├── test_model_validation.py# validação de assinatura / à-prova-de-erro
│   ├── test_engine.py          # laço, reprodutibilidade, fontes exógenas, métricas, callbacks
│   ├── test_policies.py        # políticas conhecidas + get/set_params + tags
│   └── test_forward_adp.py     # forward ADP em um problema de energia pequeno
│
└── examples/                   # capítulos do livro reescritos como consumidores da lib
    ├── ch04_diabetes_bandits.py   # "hello world": bandits, verdade oculta em ExogenousSource
    └── ch08_energy_storage_adp.py # forward ADP (figura 8.7) + replay de dados gravados
```

Divisão de responsabilidades por camada:

| Camada | Arquivos | Papel | Estabilidade esperada |
|---|---|---|---|
| **Elementos** | `elements.py` | Os dados do problema: `S_t`, `x_t`, `W_{t+1}`. | Muito alta — é o vocabulário. |
| **Problema** | `model.py`, `exogenous.py` | O contrato matemático (5 elementos) e de onde vem a incerteza. | Alta — é o "core" matemático. |
| **Estratégia** | `policies/` | Como decidir. Cresce de forma aditiva (cada nova política é um arquivo/classe a mais). | Média — cresce com o tempo. |
| **Execução** | `engine.py`, `result.py` | Rodar o jogo e resumir. | Alta. |
| **Observação** | `history.py`, `metrics.py`, `callbacks.py` | Registrar, medir (online), monitorar. Extensível pelo usuário. | Média — pontos de extensão. |
| **Infra** | `exceptions.py`, `_random.py` | Erros e reprodutibilidade. | Alta. |

Regra de dependência: `policies/`, `engine`, `metrics` dependem do núcleo (`elements`,
`model`, `exceptions`); o núcleo **nunca** importa de cima. `_random` e `exceptions` são
folhas (não importam nada do pacote).

---

## I.2. Decisões de design (ADRs)

Registros de decisão no formato *Contexto → Decisão → Consequências*. O pedido original
citava "FastAPI vs Flask, Redis vs Memcached" como exemplos — não se aplicam (esta é uma
biblioteca de modelagem numérica, sem framework web nem cache); abaixo estão os trade-offs
reais que moldaram o `pysdm`.

### ADR-01 — `Engine` (classe) em vez de `evaluate_policy` (função solta)

- **Contexto**: o repositório de curso tinha um `evaluate_policy` com 7 parâmetros
  posicionais, reimplementado em cada capítulo. Rodar de novo com outro número de episódios,
  ou trocar só a política, exigia remontar tudo.
- **Decisão**: `Engine(model, policy, horizon, ...)` guarda a configuração; `engine.run(episodes=K)`
  executa e pode ser chamado várias vezes. O nome é **`Engine`**, não `Simulator` — ele não
  só simula: prepara, valida, mede e registra. "Prepara tudo e dá o run no engine."
- **Consequências**: comparar políticas no mesmo problema é `engine.policy = outra; engine.run()`.
  `run()` reseta métricas/histórico, então cada chamada é um experimento independente. Custo:
  uma classe com estado mutável em vez de uma função pura — aceitável, o estado é a config.

### ADR-02 — Uma única base `Policy`, sem módulos PFA/CFA/VFA/DLA

- **Contexto**: o livro classifica políticas em 4 meta-classes (PFA/CFA/VFA/DLA). A tentação
  natural é criar quatro classes-base (`class CFA(Policy)`...) e quatro submódulos.
- **Decisão**: existe **um** contrato `Policy` (`decide` + hooks opcionais). As 4 meta-classes
  são apenas metadado qualitativo em `policy.tags()["policy_class"]`, não hierarquia de código.
- **Consequências**: `Engine` e `Model` nunca sabem qual "tipo" de política estão rodando —
  uma regra fechada, um argmax e uma função de valor aprendida são intercambiáveis. Políticas
  híbridas (comuns na prática, segundo o livro) não brigam com uma herança rígida. O preço é
  que a taxonomia do livro vira convenção/documentação, não é imposta pelo compilador — foi
  decisão consciente (as meta-classes são categorias de pensamento, não interfaces distintas).

### ADR-03 — `State`/`Decision`/`ExogenousInfo` declarativos (auto-dataclass)

- **Contexto**: precisávamos que o estado fosse "declarável e que se estenda" sem burocracia,
  mas também à prova de erro. Opções: `Protocol` puro (duck typing, zero base), ABC pesada
  (muito boilerplate), ou dataclass manual (usuário escreve `@dataclass` toda vez).
- **Decisão**: herdar de `sd.State` e **anotar campos**; `__init_subclass__` aplica
  `@dataclass` automaticamente e injeta `replace()`, `to_dict()`, `field_names()`. Declarar
  uma subclasse **sem campos** levanta `ModelDefinitionError` no import.
- **Consequências**: `class InventoryState(sd.State): resource: float` é tudo. `eq=False` no
  dataclass gerado porque campos podem ser arrays numpy (cujo `==` é elementwise e quebraria
  o `__eq__` gerado). Escalares seguem aceitos onde o problema é simples (a demanda de pizza é
  um `float`), então a classe declarativa é para quando `W` é um *vetor* de fontes.

### ADR-04 — `ExogenousSource` plugável, desacoplado do `Model`

- **Contexto**: a informação exógena `W` pode vir de três lugares (livro, seção 1.8.3):
  um modelo matemático, dados históricos, ou um feed real. Prender isso a `Model.exogenous_info`
  forçaria reescrever o modelo para trocar a origem dos dados.
- **Decisão**: `ExogenousSource` (ABC) é injetado no `Engine`. Default `ModelSource`
  (usa `model.exogenous_info`); `DatasetSource` (replay de sample paths gravados);
  `CallableSource` (`fn(t, state, decision) -> W`, para API/fila/sensor). Trocar a fonte não
  toca `Model` nem `Policy`.
- **Consequências**: dataset externo e feed ao vivo entram pelo mesmo ponto que a simulação.
  `DatasetSource` levanta erro explícito se os dados acabarem (mais episódios que sample paths)
  em vez de reusar dados silenciosamente — reuso silencioso invalidaria um experimento; quem
  quer reuso passa `cycle=True`.

### ADR-05 — Validação em `__init_subclass__` (falha na definição da classe)

- **Contexto**: erros comuns de modelagem (método com assinatura errada, política lendo um
  atributo de estado que não existe) apareciam como `TypeError`/`AttributeError` crípticos no
  meio de uma simulação de milhares de passos.
- **Decisão**: à prova de erro em três camadas, o mais cedo possível:
  1. `Model.__init_subclass__` checa as assinaturas dos 5 métodos **no momento da definição da
     classe** — assinatura errada → `ModelDefinitionError` mostrando a assinatura esperada.
  2. `Policy.state_requirements` declara os atributos que a política lê; o `Engine` verifica
     contra `S_0` **antes** de rodar (`MissingStateAttributeError` nomeando o que falta).
  3. O `Engine` valida o primeiro passo em runtime (objective devolve número; transition
     devolve estado novo — mutação in-place gera warning).
- **Consequências**: o feedback vem antes de qualquer trabalho pesado. Custo: introspecção de
  assinatura (`inspect.signature`) no import; implementações com `*args/**kwargs` são aceitas
  sem checagem (não há como inspecioná-las).

### ADR-06 — `numpy.random.Generator` em vez do legado `RandomState`

- **Contexto**: o código de curso usava `np.random.RandomState(base_seed + k)` por episódio —
  reprodutível, mas estatisticamente frágil (sementes correlacionadas).
- **Decisão**: um único `check_random_state` normaliza tudo para `np.random.Generator`
  (`default_rng`). O `Engine` deriva um stream independente por episódio via
  `SeedSequence.spawn` (`spawn_generators`).
- **Consequências**: reprodutibilidade fim-a-fim de um único `random_state`, e *common random
  numbers* de graça — mesmo seed + política diferente = exatamente os mesmos sample paths,
  então a comparação entre políticas é justa. `Generator` é a API recomendada pelo NumPy desde
  a 1.17.

### ADR-07 — Estado imutável (`transition` devolve estado novo)

- **Contexto**: o `evaluate_policy` de curso mutava o estado in-place. Barato, mas quebra
  lookahead (simular vários futuros a partir do mesmo `S_t`) e paralelização.
- **Decisão**: `transition` **sempre** devolve um objeto novo (idioma: `state.replace(...)`).
  Mutar e devolver o mesmo objeto gera warning explícito.
- **Consequências**: forward ADP consegue construir o estado pós-decisão sem efeito colateral;
  o `History` guarda snapshots reais (não aliases do mesmo objeto). Custo: cópia por passo —
  documentado como aceitável no v1; um modo `mutable=True` opt-in fica registrado para escala.

### ADR-08 — `Metric` streaming (online) em vez de pós-processar o DataFrame

- **Contexto**: queríamos "ver medidas ao longo do tempo, rodando" (estilo PyTorch), e um
  `Engine` genérico que não fixasse a lista de métricas.
- **Decisão**: `Metric` (ABC) com ciclo `reset() → update(record) → result()`. `result()` é
  consultável **durante** o run. Métricas built-in sempre presentes; custom via
  `engine.add_metric(...)`. `Callback` (`on_step`, `on_episode_end`) permite streaming;
  `ProgressLogger` imprime métricas correntes.
- **Consequências**: `RunResult` só agrega, não fecha a lista de medidas. Métrica pesada de
  histórico (`to_frame()` + pandas) continua possível, mas não é o único caminho — dá para
  medir sem guardar histórico (`record_history=False`) em runs gigantes.

### ADR-09 — Stepsize harmônico por célula no forward ADP (desvio da figura 8.7)

- **Contexto**: a figura 8.7 do livro usa `α_{m-1} = 1/m`, com `m` reiniciando a cada
  iteração `n`. Na prática isso faz o primeiro sample de cada iteração (α=1) **sobrescrever**
  a tabela de valor com uma trajetória ruidosa — a política aprende mal.
- **Decisão**: o default aplica `1/m` por **célula visitada** da tabela (a informação acumula
  entre iterações); `stepsize=float` permite passo constante; o `1/m` literal continua
  possível.
- **Consequências**: ~1.5× a recompensa da versão literal no exemplo do cap. 8 (medido).
  Documentado como desvio consciente e mensurado, não como "a fórmula do livro".

### ADR-10 — Build com `hatchling`, dependências pesadas como *extras*

- **Contexto**: a maioria dos usuários quer só uma política simples e não deveria ser obrigada
  a instalar pandas/matplotlib.
- **Decisão**: `hatchling` (PEP 621, sem `setup.py`, boa integração com `src/`). Dependência
  obrigatória: só `numpy`. `pandas`/`matplotlib` no extra `[analysis]`; ferramentas de
  desenvolvimento no extra `[dev]`. `to_frame()`/`plot_cumulative()` fazem *import* tardio e
  levantam `ImportError` com a instrução `pip install 'pysdm[analysis]'` se faltarem.
- **Consequências**: `pip install pysdm` é leve. O núcleo é testável sem as libs de análise.

---

## I.3. Padrões de código

### Tratamento de erros

- **Hierarquia própria com raiz única.** Tudo que a lib levanta deriva de `PysdmError`
  (`exceptions.py`), então o usuário pega problemas da biblioteca com um `except PysdmError`.
  Subclasses fazem *multiple inheritance* com o tipo built-in adequado para não surpreender
  (`ModelDefinitionError(PysdmError, TypeError)`, `MissingStateAttributeError(PysdmError,
  AttributeError)`, `InvalidDecisionError(PysdmError, ValueError)`) — código que já pega
  `TypeError` continua funcionando.
- **Falhe cedo, falhe alto.** Erro de definição de classe → no import (`__init_subclass__`).
  Erro de configuração → no construtor do `Engine`. Erro de modelagem detectável → na
  validação de `S_0` / primeiro passo, antes do trabalho pesado.
- **Mensagens acionáveis.** Toda mensagem diz *o que* está errado **e** *como* corrigir, com
  o identificador concreto. Padrão:

  ```python
  raise ModelDefinitionError(
      f"{cls_name}.{name} has signature ({got}) but pysdm will call it as "
      f"{name}({wanted}). Adjust the method to accept exactly these parameters."
  )
  ```

  Nunca deixar `KeyError`/`ValueError` genérico vazar do núcleo sem contexto.
- **Avisos para erros prováveis mas não fatais.** `transition` devolvendo o mesmo objeto usa
  `warnings.warn(..., stacklevel=2)` uma vez, não uma exceção — pode ser intencional.

### Logging

- **O núcleo não faz logging.** Uma biblioteca não deve imprimir nem configurar handlers no
  processo de quem a importa. Observação é responsabilidade de `Callback`.
- **Saída plugável.** `ProgressLogger(every=N, sink=print)` recebe o `sink` por injeção —
  passe `logger.info` para integrar ao `logging` da aplicação, ou um sink custom para um
  dashboard. O default `print` é para uso interativo (notebook).
- **Observabilidade estruturada via `History`/`Metric`**, não via texto: quem quer inspecionar
  usa `result.to_frame()` (dados) ou uma `Metric` custom (agregação), não parsing de log.

### Padrão para testes

- **`pytest` + fixtures em `conftest.py`.** Objetos de domínio reusados (`InventoryModel`,
  `ThresholdPolicy`) são fixtures; um arquivo de teste por assunto, espelhando `src/`.
- **`parametrize` para famílias.** As políticas de bandit são testadas com um único corpo
  parametrizado sobre `[Greedy, UCB, IE, Thompson]` — cobre a família sem repetição.
- **Testar invariantes matemáticas, não só valores.** Ex.: reprodutibilidade (`test_reprodu...`
  compara arrays de dois runs com mesmo seed), stepsize harmônico (`1/1` depois `1/2` leva a
  média correta), recurso nunca negativo. São propriedades, candidatas naturais a
  property-based testing (`hypothesis`) no futuro.
- **Testar o caminho à-prova-de-erro explicitamente.** Cada mecanismo de validação tem um
  teste que confirma que ele levanta a exceção certa com a mensagem certa
  (`pytest.raises(..., match="transition")`), incluindo os warnings (`pytest.warns`).
- **Exemplos como testes de integração.** `examples/ch04_*` e `examples/ch08_*` são scripts
  executáveis que exercitam a API pública ponta-a-ponta; se um capítulo não couber bem na API,
  é sinal de que a API precisa ajustar.

### Convenções gerais

- **Construtores "burros"** (convenção sklearn): `__init__` só guarda parâmetros, sem I/O nem
  validação pesada. Quem valida é o primeiro `decide()`/`run()`.
- **`get_params`/`set_params`** derivados automaticamente da assinatura do `__init__` na
  `Policy` base — habilita tuning e clonagem sem boilerplate por política.
- **`from __future__ import annotations`** em todo módulo; tipagem em toda API pública;
  docstrings Google-style com a notação do livro (`S_t`, `x_t`, `W_{t+1}`) para rastreabilidade.
- **Nomes fiéis ao Powell no núcleo** (`horizon`=T, `exogenous_info`=W); vocabulário externo
  (Gym `reset`/`step`) fica fora do núcleo.

---

## I.4. Dependências principais

Declaradas em `pyproject.toml`. Versões testadas em desenvolvimento entre parênteses.

| Dependência | Papel | Restrição | Por quê |
|---|---|---|---|
| **numpy** | Única dependência obrigatória. Arrays, `random.Generator`, quantis. | `>=1.26` (testado 2.5.0) | Base numérica; `Generator`/`SeedSequence.spawn` exigem NumPy moderno. |
| **pandas** | `result.to_frame()` — histórico como DataFrame tidy. | `>=2.0` (testado 3.0.3) | Extra `[analysis]`; import tardio. Só quem exporta dados precisa. |
| **matplotlib** | `result.plot_cumulative()` — curvas de recompensa. | `>=3.8` (testado 3.11.0) | Extra `[analysis]`; import tardio. |
| **pytest** | Suíte de testes (31 testes). | `>=8.0` (testado 9.1.1) | Extra `[dev]`. |
| **ruff** | Lint + format (substitui black/isort/flake8). | — | Extra `[dev]`. |
| **mypy** | Checagem de tipos estática. | — | Extra `[dev]`. |
| **hatchling** | Build backend (PEP 517/621). | — | `[build-system]`. |

Ambiente de referência: **Python 3.14.5** (requer `>=3.11`). Grupos de instalação:

```bash
pip install pysdm              # núcleo: só numpy
pip install 'pysdm[analysis]'  # + pandas, matplotlib (to_frame / plots)
pip install 'pysdm[dev]'       # + pytest, ruff, mypy (contribuir)
```

Regra de isolamento: o núcleo (`elements`, `model`, `policies`, `engine`, `metrics`) importa
**apenas numpy**. `pandas`/`matplotlib` só são tocados dentro de `to_frame`/`plot_cumulative`
(import tardio, `ImportError` com instrução de instalação). Isso mantém `pip install pysdm`
leve e o CI do núcleo rápido.

---

---

# Parte II — Log de design (histórico)

> O conteúdo abaixo é a proposta **pré-implementação** e o raciocínio que levou à Parte I.
> Alguns nomes mudaram na implementação (notavelmente `Simulator` → `Engine`, e os submódulos
> `policies/pfa.py|cfa.py|vfa.py|dla.py` viraram `known.py` + `adp.py` com uma base `Policy`
> única — ver ADR-01 e ADR-02). Mantido como registro de decisões e trade-offs.

## 1. Diagnóstico do estado atual

Confirmado por leitura de `common/framework.py`, `requirements.txt` e dos notebooks
`chapters/ch01_modeling/exemplo_pizza_policy (1).ipynb` e `chapters/ch04_diabetes/chapter_04.ipynb`:

- `common/framework.py` já nomeia corretamente os 5 elementos do UMF (`BaseState`,
  `BaseSimulator.observe`, `BasePolicy.decide`, `evaluate_policy`), mas:
  - `evaluate_policy` mistura simulação Monte Carlo genérica com a transition function
    como argumento solto (não há objeto que amarre `State + Decision + W + Transition +
    Objective` — os "5 elementos" nunca viram um único objeto de domínio).
  - Não existe distinção entre as 4 classes de política (PFA/CFA/VFA/DLA) — `BasePolicy`
    é uma interface única e genérica demais.
  - `np.random.RandomState` é usado diretamente, sem um padrão único de `random_state`
    reutilizável entre simulador e política.
- `chapters/ch01_modeling`: implementa o problema de inventário de pizza inteiramente com
  **funções soltas** (`T`, `C`, `simulate_demand`, `policy(r, theta_max, theta_min)`) e um
  laço `for` manual de Monte Carlo — é uma política **PFA** (threshold policy) pura, sem
  nenhuma reutilização de `common/framework.py`.
- `chapters/ch04_diabetes`: mais maduro — já usa classes (`BayesianBeliefState`,
  `DiabetesSimulator`) e várias políticas de bandit (`policy_greedy`, `policy_ucb`,
  `policy_ie`, Thompson Sampling) — **nota de reclassificação** (ver seção 4.2.1): o próprio
  livro cita o capítulo 4 (medicação de diabetes) como a ilustração de referência de **CFA**,
  não de PFA, o que exige corrigir a classificação inicial destas políticas — mas **também
  não usa** `common/framework.py`; define seu próprio `evaluate_policy` local, duplicando
  lógica. O notebook varre manualmente valores de `theta` (parâmetro de exploração) para
  achar o melhor — isto é, replica à mão o que seria um `PolicyGridSearch` de hiperparâmetros
  de política.

Conclusão do diagnóstico: os conceitos certos já existem na cabeça de quem escreveu o
código, mas cada capítulo reimplementa sua própria versão ad-hoc. O ganho de uma biblioteca
central é justamente eliminar essa duplicação e impor uma API única para os 5 elementos e
para as 4 classes de política.

---

## 2. Nome do pacote — **decidido: `pysdm`**

| Nome | Prós | Contras |
|---|---|---|
| `umf` | Curto, memorável, casa com o nome do framework do livro ("Universal Modeling Framework"). Fácil de digitar (`import umf as sd`). | Sigla muito genérica — risco de colisão de nome/conceito no PyPI e baixa "googlabilidade" (`umf` já é usado por outros projetos/siglas em outras áreas). |
| `sdam` | Mapeia 1:1 com o título do livro/repo ("Sequential Decision Analytics and Modeling"), boa rastreabilidade para quem já conhece o livro. | Menos autoexplicativo para quem não conhece o livro; pronúncia não óbvia. |
| **`pysdm`** ✅ | Deixa explícito que é Python + "Sequential Decision Modeling", termo mais usado na literatura de OR/RL do que "UMF". | "SDM" colide com outros significados conhecidos (Statistical Disclosure Control, Structural Equation Modeling em outras comunidades) — verificar disponibilidade real no PyPI/GitHub antes do release. |

Decisão do usuário: **`pysdm`**. Nas seções seguintes o nome do pacote é `pysdm`
(`import pysdm as sd`).

---

## 3. Estrutura de diretórios (`src/` layout)

```
Sequential-Decision-Analytics-Modeling/
├── pyproject.toml
├── LICENSE
├── README.md
├── ARCHITECTURE.md
├── src/
│   └── pysdm/
│       ├── __init__.py            # API pública (re-exports)
│       ├── py.typed                # marcador PEP 561 (pacote tipado)
│       ├── _typing.py              # aliases de tipo, TypeVars, Protocols internos
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state.py            # State: Protocol + ABC opcional
│       │   ├── model.py            # Model: compõe os 5 elementos do UMF
│       │   ├── random.py           # check_random_state(), utilidades de reprodutibilidade
│       │   └── exceptions.py       # hierarquia de exceções (PysdmError, InvalidStateError, ...)
│       ├── policies/
│       │   ├── __init__.py
│       │   ├── base.py             # Policy (ABC) + get_params/set_params + tags()
│       │   ├── pfa.py              # v1: ThresholdPolicy, GreedyPolicy (regra fechada, sem argmax)
│       │   ├── cfa.py              # v1 (revisão proposta, ver 4.2.1): UCBPolicy, IEPolicy,
│       │   │                       #     Thompson (argmax discreto, sem solver externo)
│       │   │                       # v2: ParametricCFA "pesado" (scipy.optimize/OR-Tools)
│       │   ├── vfa.py              # v1: TabularVFA (LinearVFA/backend neural fica para v2+)
│       │   └── dla.py              # v2: DeterministicLookahead, StochasticLookahead (rolling horizon)
│       ├── uncertainty/            # v2+/exploratório (ver 4.1.2) — Protocol de fonte de W
│       │   ├── __init__.py         #   já no v1: apenas o Protocol `ExogenousSource`
│       │   ├── historical.py       # v2+: HistoricalReplay (reamostragem de sample paths)
│       │   └── styles.py           # v2+: RegimeSwitchingProcess, SpikeProcess... (seção 1.7.4)
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── runner.py           # Simulator: evolução do evaluate_policy
│       │   └── result.py           # SimulationResult (dataclass + métricas + plot helpers)
│       ├── tuning/
│       │   ├── __init__.py
│       │   └── search.py           # PolicyGridSearch, PolicyRandomSearch (tuning de θ)
│       └── datasets/               # opcional: dados de exemplo dos capítulos (Tabela 4.1, etc.)
├── tests/
│   ├── unit/
│   │   ├── test_model.py
│   │   ├── test_policies_pfa.py
│   │   └── ...
│   └── integration/
│       └── test_simulator_end_to_end.py
├── examples/                        # ex-"chapters", agora consumidores da lib (ver seção 7)
│   ├── ch01_pizza_inventory/
│   ├── ch04_diabetes_bandits/
│   └── ...
├── docs/
│   ├── mkdocs.yml (ou conf.py, se Sphinx)
│   └── ...
└── .github/
    └── workflows/
        └── ci.yml
```

Pontos de design da estrutura:

- `src/` layout (em vez de pacote na raiz) evita importar acidentalmente a versão não
  instalada durante os testes — prática padrão recomendada por packaging.python.org e
  usada por projetos como `attrs`, `hypothesis`, `black`.
- `core/`, `policies/`, `simulation/`, `tuning/` são subpacotes separados porque têm ciclos
  de vida de estabilidade diferentes: `core` deve ser extremamente estável (é o "contrato"
  matemático), `policies` cresce com o tempo (cada nova classe de política ou variante é
  aditiva), `tuning` é opcional/plugável.
- Nomenclatura de diretórios/repositório/publicação fica fora do escopo desta discussão
  (por pedido do usuário) — o foco aqui é só a arquitetura técnica do pacote.

---

## 4. Design da API pública

### 4.1 `Model`: os 5 elementos como um único objeto

O `Model` é o "contrato" matemático do problema: estado, função de transição e função
objetivo. Ele **não** guarda a fonte de aleatoriedade "verdadeira" (isso é papel do
simulador/ambiente, mantido separado para permitir treinar/otimizar políticas contra o
`Model` e só depois avaliá-las contra um `Simulator` com a "verdade" oculta — exatamente
como `chapter_04.ipynb` já faz de forma implícita com `DiabetesSimulator` guardando
`mu_true` que a política nunca vê).

```python
import pysdm as sd
from dataclasses import dataclass

@dataclass
class InventoryState(sd.State):
    resource: float

class InventoryModel(sd.Model):
    """Compõe os 5 elementos do UMF para o problema de inventário (ch01)."""

    def __init__(self, price: float, cost: float, demand_mean: float, demand_std: float):
        self.price = price
        self.cost = cost
        self.demand_mean = demand_mean
        self.demand_std = demand_std

    def initial_state(self) -> InventoryState:
        return InventoryState(resource=0.0)

    def exogenous_info(self, state: InventoryState, decision: float, rng) -> float:
        """Amostra W^{n+1} (demanda realizada)."""
        return rng.normal(self.demand_mean, self.demand_std)

    def transition(self, state: InventoryState, decision: float, exog_info: float) -> InventoryState:
        """S^{n+1} = S^M(S^n, x^n, W^{n+1}). Retorna novo estado (imutável)."""
        new_resource = max(0.0, state.resource + decision - exog_info)
        return InventoryState(resource=new_resource)

    def objective(self, state: InventoryState, decision: float, exog_info: float) -> float:
        """C(S^n, x^n, W^{n+1})."""
        sold = min(state.resource + decision, exog_info)
        return self.price * sold - self.cost * decision
```

Decisões de design relevantes:

- `transition()` retorna um **novo** estado em vez de mutar in-place (diferente do
  `evaluate_policy` atual, que muta `state` in-place). Isso torna `Model` compatível com
  lookahead (DLA precisa simular múltiplos futuros a partir do mesmo `S^n` sem
  side-effects) e com paralelização (multiprocessing/joblib) sem race conditions.
- `State` é um `Protocol` (com uma implementação `dataclass` de conveniência), não uma
  ABC pesada — qualquer `dataclass`/`NamedTuple`/objeto do usuário que tenha os atributos
  certos funciona (duck typing estrutural, checável estaticamente com `mypy`), inspirado no
  uso de `Protocol` do próprio `typing` e no espírito "duck typing" do sklearn (que aceita
  qualquer objeto com `.fit`/`.predict`, não exige herança de uma classe base).
- `exogenous_info()` fica no `Model` como a versão "modelo" (usada por políticas CFA/DLA
  para simular cenários hipotéticos), enquanto o `Simulator`/ambiente "verdadeiro" (ver 4.3 e
  4.1.2) pode sobrescrever com uma fonte de dados real ou uma distribuição com parâmetros
  ocultos da política — igual ao padrão `mu_true` vs. crença `mu` do ch04.

### 4.1.1 Validando o design com um exemplo mais rico: Asset Acquisition (seção 2.2 do livro)

O exemplo de inventário de pizza acima usa `State`/`Decision`/`W` escalares (um único
`float` cada), o que esconde um requisito importante: o livro define os 5 elementos, em
geral, como **vetores/tuplas de componentes nomeados**. O exemplo canônico de "asset
acquisition" da seção 2.2.1 é um bom teste de estresse para a API, porque tem:

- `State` com 3 componentes de naturezas diferentes: `S_t = (R_t, D_t, p_t)` — recurso
  físico, informação determinística (demanda pendente) e preço.
- `Decision` com 2 componentes: `x_t = (x^D_t, x^O_t)` — quanto atender de demanda
  (restrito por `x^D_t ≤ R_t`, uma **restrição de factibilidade que depende do estado**) e
  quanto adquirir.
- `W` com 3 componentes: `W_{t+1} = (R̂_{t+1}, D̂_{t+1}, p̂_{t+1})` — choque exógeno de
  recurso, chegada de nova demanda e variação de preço.

```python
import pysdm as sd
from dataclasses import dataclass

@dataclass
class AssetState(sd.State):
    """S_t = (R_t, D_t, p_t): recurso disponível, demanda pendente, preço corrente."""
    resource: float   # R_t
    demand: float      # D_t
    price: float       # p_t


@dataclass
class AssetDecision(sd.Decision):
    """x_t = (x^D_t, x^O_t): quanto atender da demanda e quanto adquirir."""
    demand_served: float    # x^D_t  (restrição: x^D_t <= R_t)
    order_quantity: float   # x^O_t


@dataclass
class AssetExogenousInfo(sd.ExogenousInfo):
    """W_{t+1} = (R̂_{t+1}, D̂_{t+1}, p̂_{t+1})."""
    resource_shock: float    # R̂_{t+1}  (doações/quebras — pode ser + ou -)
    demand_arrival: float    # D̂_{t+1}
    price_change: float      # p̂_{t+1}


class AssetAcquisitionModel(sd.Model):
    """Asset acquisition problem (Powell, seção 2.2.1).

    Os 5 elementos são documentados sempre na ordem recomendada pelo livro:
    State -> Decision -> Exogenous Information -> Transition function -> Objective function.

    State variables:
        S_t = (R_t, D_t, p_t) — ver `AssetState`.

    Decision variables:
        x_t = (x^D_t, x^O_t) — ver `AssetDecision`. Restrição: x^D_t <= R_t.

    Exogenous information:
        W_{t+1} = (R̂_{t+1}, D̂_{t+1}, p̂_{t+1}) — ver `AssetExogenousInfo`.

    Transition function:
        R_{t+1} = R_t - x^D_t + x^O_t + R̂_{t+1}
        D_{t+1} = D_t - x^D_t + D̂_{t+1}
        p_{t+1} = p_t + p̂_{t+1}

    Objective function:
        C_t(S_t, x_t) = p_t * x^D_t - c_t * x^O_t
    """

    def __init__(self, acquisition_cost: float):
        self.acquisition_cost = acquisition_cost

    def initial_state(self) -> AssetState:
        ...

    def feasible_region(self, state: AssetState) -> sd.FeasibleRegion:
        """x^D_t <= R_t: restrição de factibilidade que depende do estado.

        Usado por CFA/DLA para montar o problema de otimização, e por
        `Simulator` para validar decisões produzidas por qualquer política.
        """
        return sd.Box(demand_served=(0.0, state.resource), order_quantity=(0.0, None))

    def exogenous_info(self, state, decision, rng) -> AssetExogenousInfo:
        ...

    def transition(
        self, state: AssetState, decision: AssetDecision, exog_info: AssetExogenousInfo
    ) -> AssetState:
        return AssetState(
            resource=state.resource - decision.demand_served + decision.order_quantity
                     + exog_info.resource_shock,
            demand=state.demand - decision.demand_served + exog_info.demand_arrival,
            price=state.price + exog_info.price_change,
        )

    def objective(self, state: AssetState, decision: AssetDecision, exog_info=None) -> float:
        return state.price * decision.demand_served - self.acquisition_cost * decision.order_quantity
```

O que este exemplo confirma/adiciona ao design da seção 4.1:

- **`Decision`, assim como `State`, precisa ser um `Protocol`/dataclass composto**, não um
  escalar — a API pública deve deixar isso explícito desde o v1 (o exemplo de pizza usa
  `float` só porque o problema é simples o suficiente, não porque a API assume escalares).
- **Restrições de factibilidade dependentes do estado** (`x^D_t ≤ R_t`) merecem um método
  próprio no contrato de `Model` — `feasible_region(state)` — em vez de ficarem implícitas
  dentro de cada política, evitando duplicar a lógica de "quanto no máximo posso decidir" em
  toda política nova e dando a CFA/DLA um formato estruturado para montar o problema de
  otimização (`sd.Box`, ou futuramente restrições lineares gerais). **Decidido: fica para o
  v2**, junto com CFA/DLA — só otimização precisa dele de fato; PFA/VFA tabular do v1
  garantem a restrição localmente na própria política (ex.: `x = min(theta, state.resource)`
  no `ThresholdPolicy`), mantendo o `Model` do v1 mais enxuto (`initial_state`,
  `exogenous_info`, `transition`, `objective` apenas). O exemplo `AssetAcquisitionModel`
  acima serve como referência de design para quando `feasible_region` for implementado.
- Reforça que `exogenous_info()`/`transition()`/`objective()` devem operar sobre objetos
  compostos arbitrários — a assinatura genérica `(state, decision, exog_info) -> ...` já
  suporta isso sem mudança, desde que `State`/`Decision`/`ExogenousInfo` sejam Protocols
  estruturais e não tipos fixos.
- **Convenção de documentação obrigatória**: o próprio livro recomenda ("we encourage
  readers to describe each of these five elements in this order") descrever State, Decision,
  Exogenous Information, Transition function e Objective function sempre nessa ordem ao
  modelar um problema real. Propomos adotar isso como convenção de docstring exigida (via
  template de contribuição + checagem manual em code review, não algo automatizável por
  lint) para toda subclasse de `Model` na lib e em `examples/` — inclusive um template de
  docstring pronto em `CONTRIBUTING.md` quando a implementação começar.
- O livro também nota casos com indexação dupla `S^n_t` (contador **e** tempo, ex.: hora
  dentro da semana **e** a n-ésima semana). O v1 não precisa suportar isso explicitamente,
  mas o design de `State` como Protocol estrutural já comporta essa extensão futura sem
  quebrar a API (o índice extra vira só mais um campo do `State`/do `Simulator`, não exige
  mudança na assinatura de `transition`/`objective`). Fica registrado aqui como caso de
  referência para quando os capítulos com dupla indexação (ex.: problemas de agendamento
  semanal) forem migrados.

### 4.1.2 Fontes e estilos de incerteza (seção 1.7 do livro): decompondo `exogenous_info`

O texto da seção 1.7 traz três exigências concretas para a modelagem de incerteza que o
design da seção 4.1 precisa suportar explicitamente (não são conceitos novos, mas a leitura
do capítulo torna claro que a API não pode assumir o caso mais simples):

**(a) A incerteza entra pelo `S^0` e pelo processo `W`.** Um `State` pode conter parâmetros
de uma distribuição desconhecida (ex.: `mu, beta` de uma crença Bayesiana, como
`BayesianBeliefState` no ch04) sem que isso exija nada novo do design — é só mais um campo
composto do `State`, já coberto pela seção 4.1.1.

**(b) `W_{t+1}` é, em geral, um vetor de múltiplas fontes de informação independentes**:
`W_{t+1} = (W_{t+1,i})_{i \in I_t}` (ex.: no problema de diabetes do livro, adesão à dieta,
disposição para receber injeções, perda de peso real e mudança real de glicemia são 4 fontes
de informação distintas chegando ao mesmo tempo). Isso é diretamente coberto pelo
`ExogenousInfo` composto (seção 4.1.1) — cada fonte vira um campo. O caso em que o
**conjunto `I_t` muda ao longo do tempo** (novas fontes de informação se abrem conforme a
estratégia muda) é mais raro e fica fora do escopo do v1; o `Protocol` estrutural de
`ExogenousInfo` não impede essa extensão futura, mas não vamos desenhar isso agora.

**(c) `W_{t+1}` pode depender do estado e/ou da decisão**: `W_{t+1}(S_t, x_t)` — ex.: falta
de estoque reduz demanda futura, ou comprar uma ação em grande volume aumenta seu preço
(eq. 1.26 do livro: `p_{t+1} = θ0·p_t + θ1·p_{t-1} + θ2·p_{t-2} + W_{t+1}(S_t, x_t)`, com
`W_{t+1}(S_t, x_t) = θ^x·x_t + ε_{t+1}`, `ε_{t+1} ~ N(0, |x_t|·σ_t²)`). A assinatura já
proposta em 4.1 — `exogenous_info(self, state, decision, rng)` — já cobre isso de fábrica,
sem mudança: o `state`/`decision` estão disponíveis para o autor do `Model` decidir se e como
usá-los na distribuição de `W`. Nenhuma mudança de design é necessária aqui — só reforça que
a assinatura não pode ser simplificada para `exogenous_info(rng)` (sem `state`/`decision`),
como seria tentador fazer para problemas simples tipo o de pizza.

**(d) Separar "o modelo de incerteza usado para raciocinar" da "fonte real de `W`".** A
seção 1.8.3 do livro (testagem de políticas) descreve três estratégias distintas para gerar
`W_1(ω), ..., W_T(ω)` durante uma simulação:

1. Reamostrar de **dados históricos** (ex.: combinar demandas observadas em meses diferentes
   para criar sample paths sintéticos) — não funciona bem quando `W` depende de `S_t`/`x_t`.
2. **Simular a partir de um modelo matemático** (o que `Model.exogenous_info()` já faz) —
   permite gerar quantas amostras quiser, mas o modelo pode não replicar bem correlações
   reais (entre produtos, ao longo do tempo, entre ativos).
3. **Testar em campo**, com observações reais conforme acontecem — dados reais, mas lento e
   caro (1 dia de dados novos por dia de espera).

Isso significa que `Model.exogenous_info()` (seção 4.1) deve ser entendido como apenas *uma*
implementação possível de "como gerar `W`" — a que serve de aproximação para políticas
CFA/DLA raciocinarem sobre o futuro — mas o **`Simulator`** (execução real/avaliação) precisa
poder trocar essa fonte por outra sem tocar no `Model`. Proposta: um parâmetro opcional
`exogenous_source` no `Simulator`, desacoplado do `Model`:

```python
# Estratégia 1 (padrão): usa Model.exogenous_info() com o rng do Simulator.
sim = sd.Simulator(model=inventory_model, policy=threshold_policy, n_steps=30, random_state=0)

# Estratégia 2: substitui a fonte por sample paths históricos pré-computados
# (ex.: a Tabela 1.3 do livro: uma matriz K x T de preços/demandas observados),
# sem alterar o Model nem a Policy — só como W é obtido a cada passo.
from pysdm.uncertainty import HistoricalReplay

sim_historical = sd.Simulator(
    model=inventory_model,
    policy=threshold_policy,
    n_steps=30,
    exogenous_source=HistoricalReplay(sample_paths=demand_history_df),  # shape (K, T, ...)
)

# Estratégia 3 (fora do escopo numérico da lib, mas o Protocol permite): um adaptador
# `LiveDataSource` que busca W de um feed real a cada passo, útil para testes em produção.
```

Quando `exogenous_source` não é informado, o `Simulator` usa `model.exogenous_info(...)` —
ou seja, a estratégia 2 continua sendo o caminho padrão e o mais simples de usar; as
estratégias 1 e 3 são pontos de extensão explícitos, não obrigatórios para o v1.

Além disso, a seção 1.7.4 do livro cataloga "estilos" de incerteza recorrentes (variabilidade
fina, mudanças de regime/*shifts*, rajadas/*bursts*, picos/*spikes*, eventos espaciais,
eventos sistêmicos, eventos raros, contingências). Um subpacote `pysdm.uncertainty` com
geradores reutilizáveis para alguns desses estilos (ex.: `RegimeSwitchingProcess`,
`SpikeProcess`) seria um complemento natural — mas é claramente **v2+/exploratório**: o v1
só precisa do `Protocol` de fonte de incerteza definido acima, sem implementar os estilos.
Isso é uma pergunta em aberto nova (ver seção 8).

### 4.2 Hierarquia `Policy` (PFA / CFA / VFA / DLA)

```python
from abc import ABC, abstractmethod

class Policy(ABC):
    """X^π(S^n): mapeia estado -> decisão. Análogo ao "estimator" do sklearn,
    mas o ciclo de vida é observe -> decide -> transition -> update, não fit -> predict."""

    @abstractmethod
    def decide(self, state: sd.State, model: sd.Model) -> sd.Decision:
        """Retorna x^n. `model` é passado para políticas que precisam simular
        (CFA/DLA); políticas PFA/VFA tabulares simplesmente o ignoram."""

    def update(self, state, decision, exog_info, next_state) -> None:
        """Hook opcional chamado pelo runner após cada transição, para políticas
        com estado interno aprendido (ex.: VFA atualizando V̄(S)). No-op por padrão."""

    def get_params(self) -> dict:
        """Hiperparâmetros da política (ex.: theta_min, theta_max, theta_ucb),
        no estilo sklearn `get_params`/`set_params` — habilita tuning automatizado."""

    def set_params(self, **params) -> "Policy":
        ...
```

Cada classe do livro vira um subpacote com uma responsabilidade clara.

```python
# policies/pfa.py — função analítica direta do estado, SEM nenhuma otimização embutida
class ThresholdPolicy(Policy):
    """X^π(S^n) = order-up-to rule (eq. 1.7 do livro) — regra fechada, sem argmax."""
    def __init__(self, theta_min: float, theta_max: float):
        self.theta_min, self.theta_max = theta_min, theta_max

    def decide(self, state, model=None):
        if state.resource < self.theta_min:
            return self.theta_max - state.resource
        return 0.0


class GreedyPolicy(Policy):
    """X^Greedy(S^n) = argmax_x μ̄ⁿ_x — sem termo de exploração/parâmetro; ch04."""
    def decide(self, state, model=None):
        ...


# policies/cfa.py — resolve uma otimização (mesmo que trivial/discreta) de uma versão
# parametrizada/simplificada da função objetivo, com θ ajustado por simulação
class UCBPolicy(Policy):
    """X^UCB(S^n | θ) = argmax_x [ μ̄ⁿ_x + θ·sqrt(log n / N^n_x) ] — ch04.

    Reclassificado de PFA para CFA (ver seção 4.2.1): o "argmax" sobre um bônus de
    exploração parametrizado por θ é, por definição do livro, uma cost function
    approximation — o livro cita exatamente o capítulo 4 (diabetes) como a referência
    de CFA (seção 1.8.2).
    """
    def __init__(self, theta: float = 1.0):
        self.theta = theta

    def decide(self, state, model=None):
        ...


class IEPolicy(Policy):
    """X^IE(S^n | θ) = argmax_x [ μ̄ⁿ_x + θ·σ̄ⁿ_x ] — mesma classificação de UCBPolicy."""
    def __init__(self, theta: float = 1.0):
        self.theta = theta

    def decide(self, state, model=None):
        ...


class ParametricCFA(Policy):
    """CFA de propósito geral: resolve um problema de otimização (ex.: via
    scipy.optimize/OR-Tools) usando uma aproximação paramétrica da função de custo,
    reotimizada a cada S^n. v2 — variantes "discretas" como UCB/IE acima não
    precisam de solver externo e entram no v1 (ver seção 4.2.1)."""
    def __init__(self, cost_approximation_fn, theta):
        ...
    def decide(self, state, model):
        # usa model.objective / model.transition para montar o problema de otimização
        ...


# policies/vfa.py — programação dinâmica aproximada (equação de Bellman, 1.27/1.29/1.30)
class TabularVFA(Policy):
    """X^π(S^n) = argmax_x [ C(S^n, x) + E{ V̄^{n+1}(S^{n+1}) | S^n, x } ]  (eq. 1.30).

    V̄ é uma aproximação estatística (tabular aqui) da função de valor real V, que
    nunca é computável exatamente para problemas com estado de alta dimensão.
    """
    def __init__(self, discretization, learning_rate=0.1):
        self.value_table = {}
        ...
    def decide(self, state, model):
        # argmax_x [ C(S,x,W) + V̄(S^{x}) ]
        ...
    def update(self, state, decision, exog_info, next_state):
        # atualização recursiva de V̄(S) (ex.: TD-learning / stepsize rules do cap. 6)
        ...


# policies/dla.py — decide agora otimizando sobre um modelo (tipicamente aproximado)
# que se estende por um horizonte de planejamento
class DeterministicLookahead(Policy):
    """Ex.: GPS que acha o caminho mais curto assumindo tempos de viagem conhecidos."""
    def __init__(self, horizon: int):
        self.horizon = horizon
    def decide(self, state, model):
        # constrói e resolve um problema determinístico de H períodos à frente,
        # usando model.transition/model.objective para simular o horizonte
        ...
```

O parâmetro `model` em `decide()` é opcional: `ThresholdPolicy`/`GreedyPolicy` (PFA, regra
fechada) o ignoram; `UCBPolicy`/`IEPolicy` (CFA "discreto", sem solver) também não
precisam dele; já `ParametricCFA`/`TabularVFA`/`DeterministicLookahead` usam `model` para
"olhar para dentro" da função de transição/objetivo.

### 4.2.1 PFA vs. CFA: uma fronteira mais sutil do que parece (seção 1.8.2 do livro)

Na primeira versão deste documento classificamos todas as políticas de bandit do ch04
(`policy_greedy`, `policy_ucb`, `policy_ie`, Thompson Sampling) como **PFA**. A leitura da
seção 1.8.2 mostra que isso está parcialmente errado, e a correção importa para o
empacotamento (`policies/pfa.py` vs. `policies/cfa.py`):

> "Policy function approximations (PFAs) — These are analytical functions of a state that
> directly specify an action." — regra fechada, sem otimização embutida.
>
> "Cost function approximations (CFAs) — These are policies that involve solving an
> optimization problem that is typically a simplification of the original problem, with
> parameters introduced to help make the policy work better over time. (...) We have a
> number of illustrations of CFAs later in the book (**starting with chapter 4** to learn
> the best medication for diabetes)."

Ou seja, o próprio livro usa o capítulo 4 (o mesmo problema de diabetes do nosso `ch04`) como
a ilustração de referência de **CFA**, não de PFA. Isso faz sentido observando a estrutura de
`UCBPolicy`/`IEPolicy`: ambas resolvem `argmax_x [ estimativa_x + bônus_x(θ) ]` — um
`argmax`, ainda que trivial (sobre um punhado de braços do bandit), de uma versão
**parametrizada/aproximada** do objetivo verdadeiro (que consideraria o valor de informação
completo de testar cada braço). Isso é, por definição, uma CFA. Já `ThresholdPolicy` (regra
de "order-up-to") e `GreedyPolicy` (sem termo de exploração) são PFAs "puras": avaliam uma
expressão fechada, sem nenhum `argmax`/otimização.

Critério prático para classificar uma política nova ao contribuir com `pysdm.policies`:

| Pergunta | Se sim → classe |
|---|---|
| `decide()` avalia uma expressão fechada, sem otimização/argmax nenhum? | **PFA** |
| `decide()` resolve um `argmax`/otimização — mesmo que trivial/discreta — de uma versão parametrizada/simplificada do objetivo, com θ ajustável por simulação? | **CFA** |
| `decide()` precisa de uma aproximação estatística `V̄(S)` da função de valor futura (equação de Bellman, 1.27/1.29/1.30)? | **VFA** |
| `decide()` reotimiza, a cada passo, um modelo (aproximado) que se estende por um horizonte de planejamento? | **DLA** |

Esta correção **muda o escopo recomendado do v1** (decisão 4 da tabela na seção 8):
como o exemplo "hello world" confirmado (ch04) é, pela própria definição do livro, um
showcase de **CFA**, e essa variante "discreta" de CFA (argmax sobre poucas opções, sem
solver de LP/QP) não exige nenhuma infraestrutura de otimização externa — é essencialmente
tão simples de implementar quanto uma PFA — recomendamos **revisar** o escopo do v1 para
`PFA + CFA "discreto" (sem solver) + VFA tabular`, deixando para o v2 apenas o `ParametricCFA`
"pesado" (que de fato depende de `scipy.optimize`/OR-Tools) e o `DLA`. Isso é uma pergunta
nova para o usuário confirmar (ver seção 8) — a decisão anterior ("PFA + VFA tabular") não
cobriria adequadamente o próprio exemplo "hello world" já escolhido.

O livro também é explícito que PFA/CFA/VFA/DLA são **meta-classes**: escolher a classe certa
não substitui o trabalho de desenhar a política específica dentro dela, e políticas híbridas
(que misturam elementos de mais de uma classe) são comuns na prática. A hierarquia `Policy`
proposta acomoda isso naturalmente por composição (nada impede uma política que combine, por
exemplo, um `TabularVFA` como termo de bônus dentro de um `ParametricCFA`), mas isso não
precisa ser resolvido agora — só registramos que a API não deve impor uma única herança
rígida que impeça políticas compostas no futuro.

### 4.2.2 Metadados qualitativos de política (seção 1.8.1): além da qualidade da solução

A seção 1.8.1 lista explicitamente que a escolha de uma política não depende só de
desempenho médio: **transparência** (dá para rastrear a decisão até os dados de entrada?),
**flexibilidade/adaptabilidade**, **complexidade metodológica** (a equipe consegue realmente
implementar isso?) e **requisitos de dados** também importam — e não são coisas que dá para
medir rodando um `Simulator`. Proposta: um método opcional `Policy.tags()` (inspirado em
`_get_tags()`/`_more_tags()` do sklearn, usado lá para introspecção de estimators) para
declarar esses atributos qualitativos de forma estruturada:

```python
class UCBPolicy(Policy):
    def tags(self) -> dict:
        return {
            "policy_class": "CFA",
            "needs_model": False,
            "methodological_complexity": "low",   # baixo/médio/alto, qualitativo
            "data_requirements": "online (belief state apenas)",
        }
```

Isso permite que `docs/` gere automaticamente uma tabela comparativa de políticas
disponíveis (classe, complexidade, requisitos de dados) e que `PolicyGridSearch`/relatórios
de `SimulationResult` incluam essas tags junto das métricas quantitativas — sem inventar uma
"métrica" numérica para algo que o próprio livro trata como qualitativo. Isso é aditivo e de
baixo risco (métodos default retornam `{}`); não deveria ser bloqueante para o v1, mas vale
já projetar o método na classe base `Policy` desde o início para não quebrar compatibilidade
depois.

### 4.3 `Simulator`/`Runner`: evolução do `evaluate_policy`

```python
sim = sd.Simulator(
    model=InventoryModel(price=45, cost=30, demand_mean=60, demand_std=10),
    policy=ThresholdPolicy(theta_min=80, theta_max=110),
    n_steps=30,
    random_state=42,          # int | np.random.Generator | None (ver 5)
    # exogenous_source=...,   # opcional (ver 4.1.2) — default usa model.exogenous_info()
)

result = sim.run(n_episodes=1000)   # K sample paths ω_1..ω_K (seção 1.8.3 do livro)

result.mean_reward            # F^π(S^0) ≈ (1/K) Σ_k Σ_n C(...)  — desempenho médio
result.std_reward
result.quantiles([0.05, 0.5, 0.95])   # desempenho "pior caso" (seção 1.8.1) além da média
result.mean_decision_time_ms   # custo computacional médio de policy.decide() por passo
result.max_decision_time_ms    # pior caso de tempo de execução (seção 1.8.1)
result.cumulative_by_step     # np.ndarray, shape (n_steps,)
result.plot_cumulative()      # matplotlib helper
result.to_frame()             # pandas.DataFrame long-format, para análises custom
```

Diferenças em relação ao `evaluate_policy` atual:

- `Simulator` é uma classe (não uma função com 7 parâmetros posicionais) — guarda `model`
  e `policy` como estado, permitindo reexecutar (`sim.run(n_episodes=...)`) com números de
  episódios diferentes sem reconstruir tudo, e permitindo `sim.policy = outra_politica;
  sim.run()` para comparar políticas no mesmo `Model`.
- Comparação justa entre políticas via **common random numbers**: fixar o mesmo
  `random_state` no `Simulator` e apenas trocar `sim.policy` garante que todas as políticas
  enfrentam exatamente os mesmos sample paths `ω_1..ω_K` (a mesma técnica de redução de
  variância usada implicitamente pela Tabela 1.3 do livro, onde várias políticas seriam
  comparadas sobre as mesmas 10 trajetórias de preço). Isso deve ser documentado como o
  padrão recomendado de comparação de políticas na lib, não deixado implícito.
- `exogenous_source` (opcional, seção 4.1.2) desacopla "como `W` é gerado" de `Model` e
  `Policy`, cobrindo as 3 estratégias de testagem da seção 1.8.3 (simulação via modelo,
  replay de dados históricos, feed ao vivo).
- Chama `policy.update(...)` após cada transição (necessário para VFA), o que o
  `evaluate_policy` atual não faz.
- Suporta paralelização de episódios (`n_jobs` opcional via `joblib`/`concurrent.futures`,
  no espírito do `n_jobs` do sklearn), viável porque `Model.transition()` é uma função pura.
- `SimulationResult` é um objeto rico (dataclass + métodos de plot/exportação), não um
  `dict` solto — plotting de curvas de aprendizado é uma necessidade recorrente em todos os
  capítulos (visto em ch01 e ch04), então vale a pena um helper reutilizável em vez de cada
  notebook reimplementar `plt.plot(cumulative_by_step)`. Os campos de `quantiles`/tempo de
  decisão respondem diretamente aos critérios de escolha de política da seção 1.8.1
  (qualidade média **e** pior caso; custo computacional médio **e** pior caso) — métricas que
  hoje nenhum notebook do repo calcula (só reportam a média).

### 4.4 `tuning`: formalizando o "sweep de theta" do ch04

O notebook `chapter_04.ipynb` já faz manualmente uma varredura de `theta_ie` para achar o
melhor. Isso vira um utilitário de primeira classe:

```python
from pysdm.tuning import PolicyGridSearch

search = PolicyGridSearch(
    model=diabetes_model,
    policy_factory=lambda theta: IEPolicy(theta=theta),
    param_grid={"theta": np.arange(0.0, 3.1, 0.25)},
    n_episodes=1000,
    random_state=0,
)
search.run()
search.best_params_       # {"theta": 1.25}
search.best_result_.mean_reward
search.results_           # DataFrame com uma linha por combinação de parâmetros
```

Isto é conceitualmente equivalente ao `GridSearchCV` do sklearn, mas a "validação" é feita
por simulação Monte Carlo (rodar o `Simulator` N vezes), não por cross-validation em dados
estáticos — por isso o nome é `PolicyGridSearch`, não `GridSearchCV`, para não sugerir uma
analogia que não existe (não há "fold" nem dados de treino/teste fixos, há sample paths).

---

### 4.5 `Policy` como módulo plugável: integração com RL e otimização

> Esta é a parte mais delicada do design, por pedido explícito do usuário: `Policy` é o
> ponto da API que precisa "conversar" com dois ecossistemas externos inteiros — bibliotecas
> de **otimização matemática** (scipy.optimize, OR-Tools, PuLP, cvxpy, Pyomo, solvers
> comerciais) e bibliotecas de **Reinforcement Learning** (gymnasium, stable-baselines3,
> RLlib, CleanRL, agentes custom em PyTorch/JAX) — sem que o `core` do pacote dependa de
> nenhuma delas. A resposta de design é uma arquitetura de **ports & adapters**: `Policy`
> continua sendo um contrato pequeno e estável (o "port"); toda a integração com uma
> biblioteca específica vira um **adapter** plugável por composição, não por herança.

### 4.5.1 O problema central: representações diferentes de "estado" e "decisão"

`State`/`Decision` em `pysdm` são objetos de domínio com nomes de campo com significado
(`resource`, `demand`, `price`...). Ecossistemas externos exigem outra coisa:

- Solvers de otimização (scipy.optimize, cvxpy, Pyomo, OR-Tools) trabalham com **variáveis
  de decisão tipadas do próprio solver** (`cp.Variable()`, `pyo.Var()`, `IntVar`), não com
  dataclasses do usuário.
- Bibliotecas de RL (gymnasium, stable-baselines3) trabalham com **vetores/arrays
  `numpy`** dentro de um `Space` (`Box`, `Discrete`, `Dict`) — `env.step(action)` recebe e
  devolve arrays, não objetos com nomes de campo.

Isso significa que `Policy` não pode assumir uma única representação numérica de
`State`/`Decision`. A solução é isolar essa conversão em dois Protocols pequenos e
reutilizáveis, para que qualquer adapter (RL ou otimização) os use da mesma forma:

```python
from typing import Protocol

class StateEncoder(Protocol):
    """Converte um State de domínio para a representação que a ferramenta externa espera."""
    def encode(self, state: sd.State) -> "np.ndarray": ...

class DecisionDecoder(Protocol):
    """Converte a saída bruta da ferramenta externa (vetor/ação) de volta para um Decision."""
    def decode(self, raw: "np.ndarray", state: sd.State) -> sd.Decision: ...
```

Cada `Model` concreto normalmente já sabe fazer isso para o seu próprio problema (é
específico do domínio), então a convenção é: **o autor do `Model` fornece um
`StateEncoder`/`DecisionDecoder` (ou implementa `to_vector`/`from_vector` diretamente no
`State`/`Decision`, que é o caminho mais simples para casos comuns), e os adapters de
RL/otimização abaixo os recebem via injeção de dependência.** Isso evita que `pysdm.core`
precise conhecer `numpy.ndarray` como "a" representação universal — é só a representação
usada nas fronteiras com ferramentas externas.

### 4.5.2 Integração com otimização: `SolverBackend` como estratégia plugável

CFA "pesado" e DLA resolvem um problema de otimização a cada `decide()`. Em vez de cada
política reimplementar a chamada a um solver específico, propomos um Protocol
`SolverBackend` — a política monta o problema em uma representação **solver-agnóstica**
simples (função objetivo + restrições como closures/arrays), e o backend concreto sabe
traduzir isso para a API do solver escolhido:

```python
class SolverBackend(Protocol):
    def solve(self, problem: sd.OptimizationProblem) -> sd.OptimizationResult: ...


# pysdm.contrib.optim.scipy_backend (dependência opcional: scipy, já é dependência
# core, então este backend pode inclusive viver dentro do próprio pysdm.policies)
class ScipyBackend:
    def solve(self, problem):
        from scipy.optimize import minimize, linprog
        ...  # despacha para minimize/linprog conforme problem.kind


# pysdm.contrib.optim.ortools_backend (dependência opcional: ortools)
class ORToolsBackend:
    def solve(self, problem):
        from ortools.linear_solver import pywraplp
        ...


# A policy não muda quando o backend muda:
class ParametricCFA(Policy):
    def __init__(self, cost_approximation_fn, theta, backend: SolverBackend | None = None):
        self.cost_approximation_fn = cost_approximation_fn
        self.theta = theta
        self.backend = backend or ScipyBackend()   # default sensato, sem forçar escolha

    def decide(self, state, model):
        problem = self._build_problem(state, model)   # usa model.objective/transition
        result = self.backend.solve(problem)
        return self.decision_decoder.decode(result.x, state)
```

Trocar de solver vira `ParametricCFA(..., backend=ORToolsBackend())` — nenhuma mudança na
política em si. O mesmo padrão vale para `DeterministicLookahead`/`StochasticLookahead`
(DLA), que também delegam a otimização do horizonte a um `SolverBackend`.

### 4.5.3 Integração com aprendizado de função de valor (VFA): reaproveitando a convenção `fit`/`predict`

Para VFA, o "modelo" que se quer plugável é a própria aproximação `V̄(S)` — pode ser uma
tabela (`TabularVFA`, v1), uma regressão linear, uma árvore, ou uma rede neural. Como o
sklearn já resolveu exatamente esse problema de intercambiabilidade com `fit`/`predict`,
reaproveitamos a convenção diretamente, em vez de inventar uma nova:

```python
class ValueFunctionApproximator(Protocol):
    def predict(self, features: "np.ndarray") -> float: ...
    def update(self, features: "np.ndarray", target: float) -> None: ...
    #  ^ "update" em vez de "fit" porque aqui o treino é incremental/online
    #    (um exemplo por vez, a cada transição), não em lote como o fit() clássico do sklearn


class LearnedVFA(Policy):
    """VFA genérica: aceita qualquer aproximador (tabular, sklearn, PyTorch) via composição."""
    def __init__(self, value_fn: ValueFunctionApproximator, state_encoder: StateEncoder):
        self.value_fn = value_fn
        self.state_encoder = state_encoder

    def decide(self, state, model):
        # argmax_x [ C(S,x,W) + value_fn.predict(encode(S^x)) ]
        ...

    def update(self, state, decision, exog_info, next_state):
        features = self.state_encoder.encode(next_state)
        target = ...  # ex.: TD-target
        self.value_fn.update(features, target)
```

Um adapter fino (`SklearnValueFunctionApproximator`, `TorchValueFunctionApproximator`) só
precisa mapear `predict`/`update` para `.predict()`/`.partial_fit()` do sklearn ou para um
passo de gradiente do PyTorch — o `LearnedVFA` em si nunca muda. Isso é o mesmo espírito da
decisão já registrada de manter `TabularVFA` como a única implementação do v1: os adapters
para sklearn/PyTorch entram depois, sem quebrar `LearnedVFA`.

### 4.5.4 Integração com Reinforcement Learning: `Policy` em ambas as pontas

Aqui entram dois adapters complementares, resolvendo o caso de uso citado pelo usuário
("plugar com modelos de RL"):

**(a) Usar um agente de RL já treinado como uma `Policy` do `pysdm`** (avaliação/simulação):

```python
class RLPolicyAdapter(Policy):
    """Envolve qualquer agente com uma API `predict(obs) -> action`
    (stable-baselines3, RLlib, um agente custom) como uma Policy do pysdm."""

    def __init__(self, agent, state_encoder: StateEncoder, decision_decoder: DecisionDecoder):
        self.agent = agent                     # ex.: modelo stable-baselines3 já treinado
        self.state_encoder = state_encoder
        self.decision_decoder = decision_decoder

    def decide(self, state, model=None):
        obs = self.state_encoder.encode(state)
        action, _ = self.agent.predict(obs, deterministic=True)
        return self.decision_decoder.decode(action, state)
```

Com isso, um agente treinado externamente (em qualquer framework) entra no `Simulator` como
qualquer outra `Policy` — inclusive comparável lado a lado com PFA/CFA/VFA via
`PolicyGridSearch`/common random numbers (seção 4.3).

**(b) Usar um `Model` do `pysdm` para *treinar* um agente de RL** (o caminho inverso — o
`pysdm.Model` vira o ambiente de treino):

```python
class GymEnvAdapter:
    """Expõe um pysdm.Model (+ política de geração de W) como um gymnasium.Env,
    para treinar agentes com qualquer biblioteca de RL compatível com Gymnasium."""

    def __init__(self, model: sd.Model, state_encoder, decision_decoder, random_state=None):
        ...

    def reset(self, *, seed=None, options=None):
        self._state = self.model.initial_state()
        return self.state_encoder.encode(self._state), {}

    def step(self, action):
        decision = self.decision_decoder.decode(action, self._state)
        exog_info = self.model.exogenous_info(self._state, decision, self._rng)
        reward = self.model.objective(self._state, decision, exog_info)
        self._state = self.model.transition(self._state, decision, exog_info)
        return self.state_encoder.encode(self._state), reward, False, False, {}
```

Um `GymEnvAdapter` permite treinar um agente de stable-baselines3/RLlib diretamente contra
qualquer `Model` do `pysdm`, e depois trazer o agente treinado de volta via
`RLPolicyAdapter` para avaliação com o `Simulator` (usando as mesmas métricas de qualidade,
`quantiles`, tempo de decisão etc. de qualquer outra política). Isso reconcilia a decisão já
tomada de **não** adotar o vocabulário `reset`/`step` no `Simulator` central (que continua
com a notação `S^n, x^n, W^{n+1}` do Powell) com a necessidade real de interoperar com RL: o
vocabulário Gym vive isolado num adapter opcional, não no núcleo da API.

### 4.5.5 Isolamento de dependências: `pysdm.contrib` e imports tardios

`torch`, `stable-baselines3`, `gymnasium`, `ortools`, `cvxpy` são dependências pesadas e, para
a maioria dos usuários (que só querem uma PFA/CFA simples), desnecessárias. Convenção
proposta:

- Os Protocols (`StateEncoder`, `DecisionDecoder`, `SolverBackend`, `ValueFunctionApproximator`)
  ficam no `core`/`policies` — são só interfaces, sem dependência pesada nenhuma.
- As implementações concretas que dependem de bibliotecas externas (`RLPolicyAdapter`,
  `GymEnvAdapter`, `ORToolsBackend`, `SklearnValueFunctionApproximator`,
  `TorchValueFunctionApproximator`) vivem em um namespace separado — `pysdm.contrib.rl`,
  `pysdm.contrib.optim` — com `import` tardio (dentro do `__init__`/método, não no topo do
  módulo) e uma mensagem de erro clara (`ImportError` customizado: "instale `pip install
  pysdm[rl]` para usar `RLPolicyAdapter`") se a dependência não estiver presente. Esse é o
  mesmo padrão usado por bibliotecas como `scikit-learn` (integração opcional com
  `matplotlib`/`pandas`) e `transformers` (backends opcionais de `torch`/`tensorflow`/`jax`).
- O `core` do `pysdm` nunca importa `pysdm.contrib.*` — a dependência é sempre na direção
  contrib → core, nunca o contrário, para que o núcleo permaneça leve e testável sem
  precisar instalar `torch`/`ortools`/etc. no CI do pacote base.

### 4.5.6 Caso de uso genérico: 4 políticas de naturezas diferentes, mesmo `Model`, mesmo `Simulator`

Para validar que o design realmente é genérico, eis o mesmo `InventoryModel` (seção 4.1)
sendo resolvido por 4 políticas de proveniências completamente diferentes — nenhuma delas
exige mudar o `Model` ou o `Simulator`:

```python
model = InventoryModel(price=45, cost=30, demand_mean=60, demand_std=10)

policies = {
    "PFA (regra fechada)": ThresholdPolicy(theta_min=80, theta_max=110),

    "CFA (scipy.optimize)": ParametricCFA(
        cost_approximation_fn=my_cost_approx, theta=0.9, backend=ScipyBackend(),
    ),

    "VFA (sklearn)": LearnedVFA(
        value_fn=SklearnValueFunctionApproximator(SGDRegressor()),
        state_encoder=InventoryStateEncoder(),
    ),

    "DLA/RL (agente treinado externamente)": RLPolicyAdapter(
        agent=trained_sb3_ppo_model,          # treinado via GymEnvAdapter(model), fora do pysdm
        state_encoder=InventoryStateEncoder(),
        decision_decoder=InventoryDecisionDecoder(),
    ),
}

for name, policy in policies.items():
    sim = sd.Simulator(model=model, policy=policy, n_steps=30, random_state=0)  # CRN: mesmo seed
    result = sim.run(n_episodes=1000)
    print(name, result.mean_reward, result.quantiles([0.05, 0.95]))
```

O ponto central: `Simulator` e `Model` nunca sabem (nem precisam saber) que uma política é
"só uma regra", outra "chama o scipy", outra "usa um regressor do sklearn" e outra "é uma
rede neural treinada com PPO fora do pysdm". Toda essa diferença fica encapsulada dentro de
cada `Policy`, que é o único ponto de acoplamento com o mundo externo — exatamente o
contrato que o usuário pediu para revisar com cuidado.

---

## 5. Convenções de API (inspiradas em sklearn, adaptadas ao domínio)

> Decisões confirmadas: v1 é **NumPy/SciPy puro** (sem PyTorch/JAX — `TabularVFA` cobre o
> escopo do v1; um backend neural fica para v2 caso surja demanda) e o `Simulator` mantém a
> **notação fiel ao Powell** (`S^n`, `x^n`, `W^{n+1}`), sem alinhar `decide`/`transition` à
> API `reset`/`step` do Gymnasium.


- **Construtores "burros"**: `__init__` apenas guarda parâmetros (sem side-effects, sem
  validação pesada, sem I/O) — mesma convenção do sklearn (`BaseEstimator` não valida nada
  no `__init__`, só em `fit`). Aqui, quem "valida" é o primeiro `decide()`/`run()`.
- **`get_params()` / `set_params()`**: toda `Policy` (e `Model`, quando fizer sentido)
  expõe os hiperparâmetros como atributos públicos simples e implementa `get_params`/
  `set_params`, habilitando tuning automatizado (`PolicyGridSearch`) e clonagem (`clone()`
  no estilo `sklearn.base.clone`).
- **`random_state`**: todo objeto que consome aleatoriedade (`Simulator`, `Model` quando
  amostra `W`, políticas estocásticas como Thompson Sampling) aceita `random_state: int |
  np.random.Generator | None`, resolvido internamente por um utilitário único
  `pysdm.core.random.check_random_state(...)` (mesmo padrão de
  `sklearn.utils.check_random_state`, mas devolvendo `np.random.Generator`
  (`np.random.default_rng`) em vez do legado `np.random.RandomState` usado hoje no
  `common/framework.py` — `Generator` é a API recomendada pelo próprio NumPy desde a 1.17).
- **Sem mutação in-place de `State`**: `transition()` retorna estado novo. Facilita
  debugging, replays e lookahead. Para problemas de grande escala onde copiar o estado é
  caro, o `Model` pode oferecer um modo `mutable=True` opt-in (documentado como exceção
  consciente à regra, não o padrão).
- **Serialização**: `Model` e `Policy` devem ser `pickle`-compatíveis por padrão (permite
  `joblib.dump`/paralelismo). Políticas com estado aprendido "pesado" (ex.: `TabularVFA`
  com uma tabela grande) implementam `to_dict()`/`from_dict()` além de pickle, para permitir
  salvar/carregar em formato inspecionável (JSON/YAML) — parâmetros de hiperparâmetro (θ)
  devem ser serializáveis em JSON puro sempre que possível.
- **Exceções custom**: hierarquia própria (`PysdmError` como base; `InvalidStateError`,
  `InfeasibleDecisionError`, `ConvergenceError` para VFA/CFA que não convergem), em vez de
  deixar `KeyError`/`ValueError` genéricos vazarem da lib.
- **Reprodutibilidade fim-a-fim**: um único `random_state` no `Simulator` deriva sementes
  independentes por episódio (`rng.spawn(n_episodes)` da API moderna de `Generator`, ou
  `SeedSequence`), eliminando o padrão atual `RandomState(base_seed + k)` que é reprodutível
  mas estatisticamente menos robusto que `SeedSequence`.

---

## 6. Empacotamento e ferramentas

| Aspecto | Recomendação | Observação |
|---|---|---|
| Licença | **MIT** ✅ (decidido) | `LICENSE` na raiz + `license = "MIT"` no `pyproject.toml` (PEP 639). |
| Build backend | `hatchling` (via `pyproject.toml`, PEP 621) | Simples, sem `setup.py`, boa integração com `src/` layout. Alternativa: `poetry-core`, se preferir o fluxo completo do Poetry (lockfile, `poetry publish`). |
| Versionamento | `hatch-vcs` (versão derivada de tags git) | Evita divergência entre `__version__` e tag do release; SemVer (`0.x` enquanto a API pública não estiver estável). |
| Testes | `pytest` + `pytest-cov` + `hypothesis` | `hypothesis` é especialmente valioso aqui: invariantes matemáticas (ex.: `transition()` nunca deve gerar recurso negativo, atualização Bayesiana deve sempre aumentar precisão) são ótimas propriedades para property-based testing. |
| Cobertura | `pytest-cov`, meta > 90% | Igual ao já praticado nos checklists do time. |
| Tipagem | `mypy --strict` + arquivo `py.typed` (PEP 561) | Pacote é 100% tipado desde o v1, para ser consumido com autocomplete/checagem por quem usa a lib, no espírito de `numpy`/`pandas-stubs`. |
| Lint/format | `ruff` (format + lint, incluindo regras equivalentes a `bandit` via `S` rules) | Um único binário rápido substitui `black` + `isort` + `flake8` + parte do `bandit`; `bandit` completo pode rodar em paralelo no CI se quiser cobertura extra de segurança. |
| Docs | `mkdocs-material` + `mkdocstrings` (docstrings Google-style) + `mkdocs-jupyter` (para reaproveitar os notebooks de `examples/`) | Boa renderização de LaTeX (`pymdown-extensions` `arithmatex`) para as fórmulas do UMF. Sphinx + `myst-parser`/`numpydoc` é alternativa mais "acadêmica" se preferir. |
| CI | GitHub Actions: matrix Python 3.11–3.13, jobs `lint` (ruff), `typecheck` (mypy), `test` (pytest + cobertura), `build` (`python -m build` + `twine check`) | Publicação no PyPI via "trusted publishing" (OIDC) em tag `v*`, sem tokens manuais. |
| Segurança | `bandit` + `pip-audit` no CI | Escaneamento de dependências e do próprio código. |

---

## 7. Caminho de migração dos capítulos existentes

Proposta: os capítulos (`chapters/ch01`...`ch14`) deixam de conter a lógica de domínio e
passam a ser **consumidores** da biblioteca `pysdm`, cada um definindo apenas:

1. Um `Model` específico do problema do capítulo (subclasse leve, ~20-60 linhas).
2. As políticas relevantes daquele capítulo, escolhendo entre as classes já disponíveis em
   `pysdm.policies` (ou uma subclasse pontual se o capítulo introduzir uma variante nova).
3. Narrativa/visualização no próprio notebook, chamando `sd.Simulator(...).run(...)` e os
   helpers de plot de `SimulationResult`.

Migração incremental sugerida (sem migrar tudo de uma vez):

1. **ch04 (diabetes/bandits)** primeiro — já é o mais próximo do design proposto (classes,
   múltiplas políticas, sweep de hiperparâmetro), e foi confirmado como o exemplo
   **"hello world" da documentação** do v1. Reclassificado (seção 4.2.1): é majoritariamente
   um showcase de **CFA** (`UCBPolicy`/`IEPolicy`), com `GreedyPolicy` como contraponto PFA e
   caso natural para introduzir `TabularVFA` depois. Menor distância entre "hoje" e "com a
   lib".
2. **ch01 (pizza/inventário)** segundo — mais simples (1 estado escalar, 1 decisão escalar,
   PFA de threshold), bom segundo exemplo para reforçar o padrão antes de capítulos mais
   complexos.
3. Capítulos 02-03, 05-14 são implementados **conforme o v1 da lib for cobrindo** as
   classes de política que cada um precisa (ex.: capítulos de lookahead só depois que
   `policies/dla.py` existir de fato). Cada capítulo novo é também um teste de integração
   natural da lib (se um capítulo não couber bem na API, é sinal de que a API precisa
   ajustar antes de crescer mais).

---

## 8. Decisões do usuário

| # | Pergunta | Decisão |
|---|---|---|
| 1 | Nome do pacote | **`pysdm`** |
| 2 | Backend numérico do v1 | **NumPy/SciPy puro** (sem PyTorch/JAX no v1) |
| 3 | Alinhar `Simulator` à API do Gymnasium? | **Não** — notação 100% fiel ao Powell (`S^n`, `x^n`, `W^{n+1}`) |
| 4 | Escopo de políticas do v1 | **`PFA` + `CFA` "discreto" (sem solver externo) + `VFA` tabular** (revisado — ver item 8) |
| 5 | Licença | **MIT** |
| 6 | Exemplo "hello world" da documentação | **ch04 (bandits de diabetes)** |
| 7 | `Model.feasible_region(state)` no v1? | **Não** — adiado para v2, junto com CFA/DLA |
| 8 | Reclassificação PFA→CFA (UCB/IE) e revisão do escopo do v1? | **Confirmado** — v1 = PFA + CFA discreto + VFA tabular; `ParametricCFA` (solver externo) e `DLA` ficam para v2 |
| 9 | `pysdm.uncertainty` no v1: só `Protocol` ou já geradores concretos? | **Só o `Protocol` `ExogenousSource`** — `HistoricalReplay`/`RegimeSwitchingProcess`/etc. ficam para v2+ |
| 10 | `Policy.tags()` na classe base já no v1? | **Sim**, com defaults vazios (`{}`) |

> Nota: nome do repositório GitHub, publicação em PyPI/TestPyPI e nome do diretório de
> exemplos foram explicitamente descartados da discussão a pedido do usuário — o foco é
> exclusivamente a arquitetura técnica do pacote (como as classes e módulos são desenhados,
> como o pacote é usado), não questões de nomenclatura de repositório ou distribuição.

### Novo na revisão desta seção: validado contra o texto da seção 2.2 do livro

Recebemos o texto completo da seção 2.2 ("A Universal Modeling Framework for Sequential
Decision Problems") e o exemplo de "asset acquisition", usado para validar o design contra
`State`/`Decision`/`W` compostos (seção 4.1.1). Um item novo foi introduzido no contrato de
`Model` como consequência direta desse exemplo e ainda não tinha sido confirmado
explicitamente com o usuário:

- **`Model.feasible_region(state)`**: decidido — fica para o v2, junto com CFA/DLA (ver
  item 7 da tabela de decisões acima).
- O livro também menciona o caso de indexação dupla `S^n_t` (contador + tempo). Isso **não**
  está no escopo do v1 — só deixamos registrado como algo que o design de `State` (Protocol
  estrutural) já acomoda sem mudança de assinatura, para não travar migrações futuras de
  capítulos com agendamento semanal/horário.
- O livro reforça que capítulo 9 é dedicado a aprofundar este framework — vale revisitar a
  seção 4 deste documento quando o conteúdo do capítulo 9 for incorporado ao repositório,
  já que provavelmente vai detalhar variações do UMF (múltiplos agentes, múltiplas escalas de
  tempo) que hoje não estão cobertas aqui.

## 9. Validado contra os textos das seções 1.7 e 1.8 do livro — decisões confirmadas

Recebemos os textos completos das seções 1.7 ("Modeling uncertainty") e 1.8 ("Designing
policies"), que trazem as definições formais e canônicas das 4 classes de política e da
modelagem de incerteza. Três mudanças/adições de design resultaram disso (já incorporadas
nas seções 4.1.2, 4.2, 4.2.1 e 4.2.2 acima) e já foram confirmadas com o usuário (itens 8-10
da tabela na seção 8):

**8. Reclassificação PFA → CFA e revisão do escopo do v1 — confirmado.** O livro classifica
explicitamente `UCB`/`IE` (e por extensão variantes de bandit similares) como **CFA**, não
PFA — e usa o capítulo 4 (diabetes, nosso `ch04`) como a ilustração de referência de CFA no
livro. Como essa variante "discreta" de CFA (`argmax` sobre poucas opções, sem solver de
LP/QP) é tão simples de implementar quanto uma PFA, o escopo do v1 passa de `PFA + VFA
tabular` para **`PFA + CFA "discreto" + VFA tabular`**, adiando para v2 apenas o
`ParametricCFA` que de fato depende de solver externo, além do DLA.

**9. `pysdm.uncertainty` como subpacote de v2+ — confirmado.** O v1 define apenas o
`Protocol` de fonte de incerteza (`ExogenousSource`, usado pelo `Simulator` via o parâmetro
opcional `exogenous_source`), deixando implementações concretas (`HistoricalReplay`,
`RegimeSwitchingProcess`, etc.) para v2+.

**10. `Policy.tags()` — confirmado para o v1.** Entra na classe base `Policy` já no v1, com
implementação default retornando `{}` (baixo risco, aditivo, evita quebra de compatibilidade
quando a introspecção qualitativa for expandida no futuro).
