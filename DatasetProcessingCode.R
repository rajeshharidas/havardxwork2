##########################################################
# Create Census pay train set, and validation set
##########################################################

# Note: this process could take a couple of minutes

if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(dply))
  install.packages("dply", repos = "http://cran.us.r-project.org")
if (!require(gridExtra))
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(kableExtra))
  install.packages("kableExtra", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(gridExtra)
library(kableExtra)


# Adult Census Income
# https://www.kaggle.com/uciml/adult-census-income

#download the dataset from the staging github location
dl <- tempfile()
download.file("https://github.com/rajeshharidas/havardxwork2/raw/main/adult.csv.zip",
              dl)

#read all the data into R dataset
adultpay <-
  fread(
    text = gsub(",", "\t", readLines(unzip(dl, "adult.csv"))),
    col.names = c(
      "age",
      "workclass",
      "fnlwgt",
      "education",
      "education.num",
      "marital.status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "capital.gain",
      "capital.loss",
      "hours.per.week",
      "native.country",
      "income"
    )
  )

#Keep only USA data
#Remove '?' from the work class and rename it to class, and finally remove workclass
#Rename all columns with a '.' in it
#Remove capital gain and loss column
#remove non-alphanumeric character from column data
#rename the label for below and above 50K income
adultpayclean <-
  adultpay %>% filter (native.country == 'United-States') %>%
  mutate (class = ifelse(workclass == '?', 'Unknown', str_replace_all(workclass, "-", ""))) %>%
  select(-workclass, -capital.gain, -capital.loss) %>%
  rename(
    c(
      eduyears = education.num,
      maritalstatus = marital.status,
      hoursperweek = hours.per.week,
      native = native.country
    )
  ) %>%
  mutate (maritalstatus = ifelse(
    maritalstatus == '?',
    'Unknown',
    str_replace_all(maritalstatus, "-", "")
  )) %>%
  mutate (occupation = ifelse(
    occupation == '?',
    'Unknown',
    str_replace_all(occupation, "-", "")
  )) %>%
  mutate (education = ifelse(education == '?', 'Unknown', str_replace_all(education, "-", ""))) %>%
  mutate (relationship = ifelse(
    relationship == '?',
    'Unknown',
    str_replace_all(relationship, "-", "")
  )) %>%
  mutate (native = ifelse(native == '?', 'Unknown', str_replace_all(native, "-", ""))) %>%
  mutate (income = ifelse(
    income == '?',
    'Unknown',
    str_replace_all(income, "<=50K", "AtBelow50K")
  )) %>%
  mutate (income = ifelse(
    income == '?',
    'Unknown',
    str_replace_all(income, ">50K", "Above50K")
  ))

# R 4.0 or later:
#convert all the character labels to factors
adultpayclean <-
  as.data.frame(adultpayclean) %>% mutate(
    education = as.factor(education),
    maritalstatus = as.factor(maritalstatus),
    occupation = as.factor(occupation),
    relationship = as.factor(relationship),
    race = as.factor(race),
    sex = as.factor(sex),
    class = as.factor(class),
    income = as.factor(income)
  )


# Validation set will be 10% of adultpay data
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <-
  createDataPartition(
    y = adultpayclean$income,
    times = 1,
    p = 0.1,
    list = FALSE
  )
adultpayclean_train <- adultpayclean[-test_index, ]
adultpayclean_validation <- adultpayclean[test_index, ]

glimpse(adultpay)


glimpse(adultpayclean)

dim(adultpayclean)
dim(adultpayclean_train)
dim(adultpayclean_validation)

summary(adultpayclean)
summary(adultpayclean_train)

tribble(
  ~"Dataset",     ~"Number of Rows",    ~"Number of Columns",
  #--             |--                   |----
  "train",          nrow(adultpayclean_train),        ncol(adultpayclean_train),
  "validation",   nrow(adultpayclean_validation),     ncol(adultpayclean_validation)
) %>% knitr::kable() %>% kable_styling(bootstrap_options = c("striped", "hover", "condensed"))
