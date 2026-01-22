# Проект: Сборный проект - 2

## 1. Описание исследования

## 2. Цель исследования

Оптимизация управления персоналом бизнеса: на основании предоставленных данных предложить бизнесу, как избежать финансовых потерь и оттока сотрудников.

## 3. Задачи исследования

*   Построить модель, которая сможет предсказать уровень удовлетворённости сотрудника на основе данных заказчика.
*   Построить модель, которая сможет на основе данных заказчика предсказать то, что сотрудник уволится из компании.

## 4. Исходные данные

*   `id` — уникальный идентификатор сотрудника;
*   `dept` — отдел, в котором работает сотрудник;
*   `level` — уровень занимаемой должности;
*   `workload` — уровень загруженности сотрудника;
*   `employment_years` — длительность работы в компании (в годах);
*   `last_year_promo` — показывает, было ли повышение за последний год;
*   `last_year_violations` — показывает ли сотрудник нарушал трудовой договор за последний год;
*   `supervisor_evaluation` — оценка качества работы сотрудника, которую дал руководитель;
*   `salary` — ежемесячная зарплата сотрудника;
*   `job_satisfaction_rate` — уровень удовлетворённости сотрудника работой в компании, первый целевой признак.
*   `qiit` - увольнение сотрудника из компании, второй целевой признак.

## 5. Этапы исследования

*   Часть 6. Загрузка библиотек:
    *   6.1. Загрузка библиотек.
*   Часть 7. Изучение общей информации:
    *   7.1. Загрузка и изучение файлов с данными, получение общей информации.
    *   7.2. Нахождение и ликвидация пропусков.
    *   7.3. Итоги изучения общей информации.
*   Часть 8. Предобработка данных:
    *   8.1. Удаление явных дубликатов.
    *   8.2. Приведение данных к нужным типам.
    *   8.3. Анализ уникальных значений в признаках. Удаление неявных дубликатов
    *   8.4. Итоги предобработки данных.
*   Часть 9. Исследовательский анализ данных:
    *   9.1. Нахождение и исправление аномалий и ошибок.
    *   9.2. Построение и анализ распределений, диаграмм рассеяния.
    *   9.3. Анализ матрицы корреляции.
    *   9.4. Итоги исследовательского анализа.
*   Часть 10. ЗАДАЧА 1: ПРЕДСКАЗАНИЕ УРОВНЯ УДОВЛЕТВОРЕННОСТИ СОТРУДНИКА РАБОТОЙ КОМПАНИИ:
    *   10.1. Обучение моделей. Выбор лучшей.
    *   10.2. Оценка качества модели на SMAP.
    *   10.3. Выводы об обучению моделей.
*   Часть 11. ЗАДАЧА 2: ПРЕДСКАЗАНИЕ УВОЛЬНЕНЯ СОТРУДНИКА ИЗ КОМПАНИИ:
    *   11.1. Добавление нового входного признака.
    *   11.2. Обучение модели.
    *   11.3. Оценка качества модели.
    *   11.4. Выводы об обучении модели.
*   Часть 12. Общий вывод:
    *   12. Общий вывод.

## Загрузки:
python

!pip install phik==0.11.1 -q

!pip install shap==0.40.0 -q

!pip install scipy==1.10.1 -q

!pip install -U scikit-learn -q

!pip install --upgrade scikit-learn lightgbm -q


## Используемые библиотеки:

python

import re

import math

import shap

import pandas as pd

import phik

import pickle

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import warnings

from phik import resources

from phik.report import plot_correlation_matrix

from scipy import stats as st

from scipy.stats import binom, shapiro, loguniform, ttest_ind

from sklearn.preprocessing import (OneHotEncoder,

                                   OrdinalEncoder,
                                   
                                   StandardScaler,
                                   
                                   MinMaxScaler,
                                   
                                   LabelEncoder,
                                   
                                   FunctionTransformer)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Ridge

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor, DummyClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import StackingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (make_scorer,

                             accuracy_score,
                             
                             roc_auc_score,
                             
                             recall_score,
                             
                             f1_score,
                             
                             mean_absolute_error,
                             
                             r2_score,
                             
                             precision_score,
                             
                             confusion_matrix)
```
