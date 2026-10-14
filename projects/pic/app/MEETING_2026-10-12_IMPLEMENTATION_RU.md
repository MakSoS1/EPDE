# EPDE Explorer: итоги созвона 12.10.2026 — реализация 14.10.2026

**Исследовательская ветка:** `research/nir1-mdl-factorial-2026-10-14`.  
**Остальные ветки:** `nir1`, `main`, оригинальный ITMO EPDE не изменялись этим этапом.  
**Веб-деплой:** пользователь выполняет сам после проверки GitHub Actions.  
**Входная точка:** `projects/pic/app/Home.py`.

## Реализованные изменения

| Замечание встречи | Реализация | Код |
| --- | --- | --- |
| Лишний свободный коэффициент на странице Signal check | По умолчанию diagnostic fit без intercept. Отдельный явный toggle для свободного диагностического коэффициента. Исправлен R², чтобы не скрывать постоянный bias через дисперсию остатков. | `epde_bench/truthcheck.py`, `views/2_Signal_check.py` |
| Signal check и Derivatives кажутся одинаковыми | Явные определения разных задач: проверка наблюдаемости *известного* закона и оценка ошибки численного дифференцирования. | `views/2_Signal_check.py`, `views/3_Derivatives.py` |
| Настроить степень и окно polynomial derivatives | UI для Chebyshev poly_degree + odd polynomial_window в поиске, в исследовании производных и собственных данных. Общая строгая проверка degree >= max derivative, window >= degree+2, минимальных размеров сетки. | `epde_bench/poly_options.py`, `epde_bench/preprocessing.py`, `app/support/options.py`, `views/3_Derivatives.py`, `views/6_Your_data.py` |
| Нужен токен 1/x для PDEDivide | Реализован опциональный фиксированный CustomTokens 1/x по пространственной координате. Без eval и произвольного кода. На нулевой координате выбрасывается ошибка до поиска. | `epde_bench/tokens.py`, `app/support/options.py`, `views/6_Your_data.py` |
| Импорт / экспорт параметров | YAML с safe_load, ограничением 64 KiB, allowlist семейств токенов, бюджетов; экспорт воспроизводимого `search:` override. | `app/support/config_io.py`, `app/support/options.py`, `views/6_Your_data.py` |
| Отдельный режим SINDy | Встроена страница SINDy для запуска реально существующего PySINDy-адаптера и EPDE на одинаковых record/noise/seed. Сравнение через существующие Results и Benchmark. | `views/4_SINDy.py`, `app/Home.py` |
| Бенчмарк разных фреймворков | В Benchmark уточнена сопоставимость EPDE и PySINDy и ограничено число параллельных воркеров одного задания до 2 для демонстрации. | `views/8_Benchmark.py` |
| Названия PIC | Продукт называется EPDE Explorer; исторический путь `projects/pic` сохранён, чтобы не сломать существующие тесты, импортируемые пути и артефакты. | `app/Home.py`, документация |

## Отдельно: что научрук предлагал на будущее, а не требовал немедленно

- **LLM discovery:** внешний LLM-адаптер, платные API, защита ключа и квоты. Для этого ещё не выбран API, benchmark протокол и механизм авторизации. Демо не отправляет LLM-запросы и не выставляет никакой секрет наружу. Реальную платную интеграцию нельзя считать сделанной.
- **Остальные PDEComp-фреймворки:** согласовать с Владиславом точный список и единый протокол входных данных, допустимых библиотек, результатов и метрик. Сайт уже поддерживает EPDE и PySINDy, но не все фреймворки.
- **Полноценный общедоступный сервис:** нужны авторизация, изоляция загрузок и заданий, глобальные квоты и постоянное хранилище. Текущий Streamlit Community Cloud — лабораторное shared-instance demo. Ограничение 2 воркера в одной кампании **не является** полноценным rate limit.
- **Спектральное/нейросетевое дифференцирование:** на созвоне их специально предложили отложить.
- **Полное переименование projects/pic:** требовало бы изменения API-путей и артефактов существующих тестов, не является исправлением научной функциональности.

## Тесты

- `test_streamlit_supervisor_20261012.py`: intercept, смещение R², неправильная степень/окно, YAML инъекции, запуск страниц.
- `test_streamlit_reciprocal_token.py`: создание токена и **реальный короткий EPDE-поиск** с 1/x на области без нуля.
- Существующие Streamlit regression suites: `test_app.py`, `test_app_interfaces.py`, `test_app_inputs.py`.
- Запуск PySINDy как реальной команды через `bench.py run ... --variant pysindy`.
- Контрольный workflow `nir1-streamlit-postcall-qa.yml` (Python 3.12, точный `uv.lock`, `cpu + app`).
- Второй workflow `nir1-streamlit-cloud-requirements.yml` (Python 3.13, **точный app/requirements.txt** для Community Cloud).

**Успешная исследовательская проверка:** https://github.com/MakSoS1/EPDE/actions/runs/38614535882

## Деплой вручную (делает владелец)

1. Проверить успешность обоих CI workflows.
2. В Streamlit Community Cloud выбрать существующий `MakSoS1/EPDE` и ветку `research/nir1-mdl-factorial-2026-10-14`.
3. Путь приложения: `projects/pic/app/Home.py`; Python 3.13.
4. Проверить два состояния Signal check intercept ON/OFF; polynomial degree=3 с запросом производной 4-го порядка (должна быть **ошибка**); degree=5/window=11 (должно работать).
5. Для 1/x использовать только поле с координатами x в интервале, исключающем ноль. Проверить запуск PySINDy и EPDE на одних данных.
6. Скачать JSON результатов, проверить импорт/экспорт YAML и хранить ценные результаты вне Community Cloud, где они могут исчезнуть при перезапуске.
7. Для публичного широкого доступа сначала внедрить multi-tenant isolation и quotas.

**До запуска реального сервера Streamlit Cloud нельзя обещать отсутствие проблем с конкретным хостингом.**
