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

if (!require(dplyr))
  install.packages("dplyr", repos = "http://cran.us.r-project.org")

if (!require(gridExtra))
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")

if (!require(kableExtra))
  install.packages("kableExtra", repos = "http://cran.us.r-project.org")

if (!require(epiDisplay))
  install.packages("epiDisplay")


library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(gridExtra)
library(kableExtra)
library(epiDisplay)

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

#function to get the mode for categorical column. This is used to impute missing values
getmode <- function(v){
  v=v[nchar(as.character(v))>0]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#Keep only USA data
#Remove '?' from the work class and rename it to class, and finally remove workclass
#Rename all columns with a '.' in it
#Remove capital gain and loss column
#remove non-alphanumeric character from column data
#rename the label for below and above 50K income
#impute ? values to the modes in categorical columns
adultpayclean <- adultpay %>% filter (native.country == 'United-States') %>%
  mutate (class = ifelse(workclass == '?', getmode(adultpay$workclass), str_replace_all(workclass, "-", ""))) %>%
  dplyr::select(-workclass, -capital.gain, -capital.loss) %>%
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
    getmode(adultpay$maritalstatus),
    str_replace_all(maritalstatus, "-", "")
  )) %>%
  mutate (occupation = ifelse(
    occupation == '?',
    getmode(adultpay$occupation),
    str_replace_all(occupation, "-", "")
  )) %>%
  mutate (education = ifelse(education == '?', getmode(adultpay$education), str_replace_all(education, "-", ""))) %>%
  mutate (relationship = ifelse(
    relationship == '?',
    getmode(adultpay$relationship),
    str_replace_all(relationship, "-", "")
  )) %>%
  mutate (native = ifelse(native == '?', 'Unknown', str_replace_all(native, "-", ""))) %>%
  mutate (income = ifelse(
    income == '?',
    getmode(adultpay$income),
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

tab1(adultpayclean$education, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(education) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(education,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$race, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(race) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(race,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$maritalstatus, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(maritalstatus) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(maritalstatus,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$sex, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(sex) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(sex,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$relationship, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(relationship) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(relationship,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$class, sort.group = "decreasing", cum.percent = TRUE)
adultpayclean %>% group_by(class) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(class,col=income,fill=income)) + scale_y_sqrt()

tab1(adultpayclean$income, sort.group = "decreasing", cum.percent = TRUE)

adultpayclean %>% group_by(age) %>%
  mutate(n=n())  %>% ggplot()  +
  geom_bar(aes(age,col=income,fill=income)) + scale_y_sqrt()






