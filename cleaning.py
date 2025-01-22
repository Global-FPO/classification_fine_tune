import nltk
#mport unidecode
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords=set(stopwords.words('english'))
#import contractions
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt_tab')
 
 
def text_handle(col):
 
  # Convert to lowercase
  text = col.lower()
 
  # Expand Contractions
  #text = contractions.fix(text)
 
  text = text.replace("#", " ")
  text= text.replace('*',' ')
  text = text.replace("/", " ")
  text= text.replace(':',' ')
  text = text.replace("-", " ")
  text= text.replace('--',' ')
  text = text.replace(";", " ")
  text= text.replace('.',' ')
 
  # Remove URLs
  text = re.sub(r'http\S+', '', text)
 
  # Remove numbers
  text = re.sub(r'\d+', '', text)
 
  # Remove punctuation
  #text = re.sub(r'[^\w\s]', '', text)
 
 
  # Remove extra spaces
  text = re.sub(r'\s+', ' ', text).strip()
 
  # Remove accents
  #text = unidecode.unidecode(text)
 
  # Remove stopwords
  text = ' '.join([word for word in text.split() if word not in stopwords])
 
  # Tokenize words
  words = word_tokenize(text)
 
  # Stem words
  #stemmer = SnowballStemmer('english')
  #words = [stemmer.stem(word) for word in words]
 
  # Join words back into a single string
  text = ' '.join(words)
 
  return text
 
 