# Протокол проверки multisample

13 ноября 2026 года. Исходная версия: `8690035e90b234b34ed5a2fb0bb8e4a814dcbdf7`, `ITMO-NSS-team/EPDE`, ветка `domain_refactor`. Исходная версия проверялась в отдельном неизменённом checkout; доработанная — в `research/multisample-2026-11-13`.

## Результаты

| Проверка | Результат | Время |
|---|---|---:|
| Полный `tests/unit`, исходная версия, зависимости установлены | **2603 passed, 5 failed**, 0 errors/skips | 394.96 s |
| Полный `tests/unit`, итоговая версия | **2703 passed, 5 failed**, 0 errors/skips | 373.33 s |
| Новые специализированные тесты multisample, входят в итоговый unit-прогон | **96 passed** | В составе общего прогона |
| Автоматические AST-проверки четырёх новых тестовых модулей | **4 passed** | В составе общего прогона |
| `projects/pic/tests`, исходная версия | **88 passed, 23 subtests passed** | 9.98 s |
| `projects/pic/tests`, итоговая версия | **88 passed, 23 subtests passed** | 9.95 s |
| Пример из русского отчёта, реальные FD-производные и optimizer | Выполнен; коэффициент `-0.3999999995117188` вместо `-0.4` | Лог сохранён |
| Численная матрица benchmark | **97 ok, 1 no_candidate, 0 errors, 0 timeouts**, всего 98 | 746.4 суммарных worker seconds |
| Обычный `pytest` по всему репозиторию | **3 ошибки сбора**, выполнение общей сессии прерывается | 3.81 s |
| `pytest tests/system` | 0 собранных тестов, exit 5 | Не считается успешным тестированием |

Исходные и итоговые JUnit XML сравнивались по полным именам тестов. Для всех 2608 исходных случаев статус совпал. Прирост в 100 случаев состоит из 96 специальных регрессий и четырёх автоматически добавленных проверок новых файлов существующим `test_collection_health.py`. Новых падений unit-тестов не обнаружено. Это не означает, что весь исходный репозиторий имеет зелёный pytest: пять падений и три ошибки общего сбора перечислены ниже.

Один существующий тест `test_row_budget.py` уточнён вместе с исправлением постоянства: проверяется исключение константного токена из текущего пула и сохранение исходного объявления. Прежнее ожидание мутации переданного семейства было частью исправленной ошибки жизненного цикла.

Машинный протокол, версии и SHA-256 файлов доказательств: [verification.json](results/2026-11-13/verification/verification.json).

## Пять исходных падений

Все пять находятся в `test_lr_free_solve.py` и сравнивают точные SHA-256 отпечатки результатов с историческими значениями:

1. `TestTheShippedSolveIsPinned::test_the_prediction_and_loss_are_bitwise_unchanged[adam+lbfgs]`
2. `TestTheShippedSolveIsPinned::test_the_prediction_and_loss_are_bitwise_unchanged[lbfgs-only]`
3. `TestTheS1PathIsPinned::test_the_affine_input_solve_is_unchanged`
4. `TestThePeriodicEmbedding::test_it_is_inert_without_a_periodic_axis`
5. `TestTheScaleInvariantObservationLoss::test_the_default_path_is_the_s1_pin`

| Вычисление | Исторический ожидаемый hash | Наблюдаемый hash на обеих версиях |
|---|---|---|
| Adam + L-BFGS | `6f9cc57457ed66c62a35fddb` | `51405a754d1aa60d0519935c` |
| Только L-BFGS | `361e35ab2ff3ef152e604911` | `eb998099465c55ba0d820e55` |
| S1 / affine input, три проверки | `c28c0dc36d3b02622e576293` | `fe51316668985afa067e7120` |

Отдельный повтор этих пяти проверок на изменённой версии дал те же hashes; проверка побитовой воспроизводимости повторных вычислений внутри одного процесса прошла. Первые две проверки также прошли утверждения о конечности предсказаний и loss перед сравнением hash.

Таким образом, установлено, что расхождения существовали до изменений multisample. Конкретная зависимость или особенность платформы, объясняющая отличие от исторических значений, не изолирована. Эталонные hashes не заменялись и проверки не ослаблялись ради зелёного результата.

## Общий сбор pytest

Одинаковые три ошибки воспроизведены на исходной и изменённой версиях:

| Источник | Причина |
|---|---|
| `projects/pic/data/dp/fno_test.py` | При импорте загружается отсутствующий `./dp.npz` |
| `projects/robot/epde_u/test_invariant_terms.py` | Импортируемый production-модуль поставлен только отдельным неинтегрированным patch |
| `tests/unit/conftest.py` и `tests/system/conftest.py` | Оба регистрируют аргумент `--path`; совместный сбор даёт конфликт |

Общий `pytest -q --timeout=90` действительно выполнен и завершился на сборе. Его результат не подменён успешным запуском выбранного поднабора. Основной unit-набор и весь самодостаточный PIC-набор выполнены отдельно, полностью. Каталог `tests/system` при стандартном поиске не собирает тестов; старые CLI-примеры не засчитаны как прошедшие тесты. Исследовательские данные не подделывались, robot patch не применялся в рамках multisample.

## Покрытие новых регрессий

| Модуль | Случаев | Что проверяется |
|---|---:|---|
| `test_multisample_api.py` | 20 | Конфигурация, сериализация, неверные сочетания, пустые/несогласованные данные, ID, адаптер, полный optimizer, повторный fit и замена данных |
| `test_multisample_domain.py` | 35 | 2-D области, stride, time axis, порядок строк, независимые FD, rebuild/smoothing, границы, кэш, веса, evaluators, постоянные параметры и смена активного набора |
| `test_multisample_fit.py` | 37 | Общий ранг, общий вектор, отрицательные контроли, плотность, физический масштаб, перестановка записей, intercept, finite checks, четыре discrepancy и фактическая instability, plain/assignment |
| `test_multisample_fill.py` | 4 | Согласованный взвешенный Gram предложений для strong/weak × samples/points |

Ошибки воспроизводились до исправлений. Сохранены начальные RED-логи API, общей регрессии, geometry, configuration guard, fill-and-prune и constancy. Повторные GREEN-проверки, финальный полный прогон и реальные жизненные циклы адаптера доступны рядом с ними. Фиксация локально вырожденной, но совместно идентифицируемой структуры не подменяет измерения полного эволюционного поиска — последний представлен отдельными 60 заданиями benchmark.

## Среда

Python **3.12.14**, Linux x86-64, CPU. Основные версии:

| Пакет | Версия |
|---|---|
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| Torch | 2.15.0+cpu |
| DeepXDE | 1.15.1 |
| pytest | 9.2.0 |
| pytest-timeout | 2.4.0 |
| PySINDy | 2.1.0 |
| derivative | 0.6.3 |

[requirements-tested.txt](requirements-tested.txt) фиксирует основные зависимости. Это запись проверенной среды, а не обещание одинаковых битов на любой платформе и не полный lockfile транзитивных пакетов.

Первый пробный unit-прогон в неполной среде получил 125 ошибок/падений, связанных с отсутствием DeepXDE. Для сравнения использован повторный прогон чистой исходной версии после установки зависимости; эти ошибки не записаны в регрессии multisample.

Для PIC сначала отсутствовал PySINDy. Версия 2.0.0 выявила несовместимость с приватной функцией `_preprocess_data` scikit-learn 1.8: ожидалось пять результатов вместо шести. PySINDy 2.1.0 содержит собственную совместимую реализацию, после чего 88 тестов прошли на обеих версиях EPDE. Добавлены только PySINDy, derivative, importlib-metadata и zipp; сравнение снимков установленных пакетов подтвердило, что существующие зависимости численного стека не менялись.

## Команды воспроизведения

Из корня проверяемой ветки, в отдельном окружении с Python 3.12:

```bash
python -m venv .venv-multisample
source .venv-multisample/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install -r projects/multisample/requirements-tested.txt
python -m pip install -e . --no-deps
export DDE_BACKEND=pytorch
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest tests/unit -q --timeout=90 --junitxml=unit.xml
python -m pytest projects/pic/tests -q --timeout=90 --junitxml=pic.xml
python -m pytest -q --timeout=90
python projects/multisample/benchmark.py --check-data
python projects/multisample/benchmark.py --suite all --workers 3 --output projects/multisample/results/reproduction/benchmark.json
```

Не запускайте независимые кампании в один и тот же каталог результатов. Длительность не является строгим сравнением производительности: машины, фоновые процессы и расширение популяции могут влиять на неё. Важно сохранить одинаковые бюджеты A/B, все seeds и статусы исходов.

Кампания 98 заданий была выполнена до заключительной правки проверки постоянства. После неё реально перестроены 16 различных конфигураций пулов со всеми доступными признаками; удаляемых новой проверкой столбцов не найдено. Последующий manifest исходников записан отдельно и не выдан за ретроспективный отпечаток более раннего исполнения. Его 153 файловых hash совпадают с итоговым production-кодом и runner.

В исходных metadata кампании строка описания discovery содержала слово `unseeded`. Это неточная формулировка описания, а не фактическое поведение: runner явно устанавливает `random.seed`, `np.random.seed`, Torch seed и `PYTHONHASHSEED` из записанного seed; истинные уравнения в начальную популяцию не вставляются. Формулировка исправлена для будущих запусков, исходные метаданные сохранены.

## Основные доказательства

- [Исходный unit-прогон](results/2026-11-13/verification/baseline-complete-env.log) и [JUnit XML](results/2026-11-13/verification/baseline-complete-env.xml).
- [Итоговый unit-прогон](results/2026-11-13/verification/final-unit.log) и [JUnit XML](results/2026-11-13/verification/final-unit.xml).
- [Повтор пяти исторических расхождений](results/2026-11-13/verification/feature-historical-pin-comparison.log).
- [PIC на исходной версии](results/2026-11-13/verification/baseline-pic-project-final.log) и [PIC на итоговой версии](results/2026-11-13/verification/pic-project-final.log).
- [Фактический общий pytest](results/2026-11-13/verification/repository-pytest-final.log) и [сбор исходной версии](results/2026-11-13/verification/baseline-repository-collection.log).
- [Пример из отчёта](results/2026-11-13/verification/report-example.log).

Повторная проверка кода выявила два существенных дефекта: перезапись коэффициентов при joint assignment + LASSO и ошибочное/разрушительное определение постоянства. Оба исправлены с регрессиями и входят в итоговый прогон. Остальные ограничения методов и метрик перечислены в [основном русском отчёте](REPORT_RU.md).
