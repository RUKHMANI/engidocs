# Implementation from https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk
import nltk
import pickle

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


def file_read(fname):
        text_str = " "
        text_list = []
        with open (fname, "r") as myfile:
                text_list += myfile.readlines()
        for lis in text_list:
            text_str += lis
        return text_str

'''
text_str = HOSPITAL NAME

OPERATIVE REPORT

Name:PATIENT MR#: 000000
Location:LOC Acct# :01
DOB:06/18/2013
Admit Date: Discharge Date:



OPERATIVE REPORT


SURGEON: ABC, MD
1ST ASSISTANT: Dr. Doc2.
2ND ASSISTANT:
ANESTHESIA: General.
DATE OF SURGERY: 10/10/2017


PREOPERATIVE DIAGNOSES: Bilateral cryptorchidism.

POSTOPERATIVE DIAGNOSIS: Bilateral cryptorchidism with right appendix testis and appendix
epididymis.

PROCEDURE PERFORMED: Sequential right orchiopexy, excision of right appendix testis and
appendix epididymis.

ANESTHESIOLOGIST: Doc3, MD

INDICATION FOR OPERATION: This is a 4-year-old boy, who was admitted because of bilateral undescended testes, from which a sequential bilateral orchiopexy will be done at this time. The right side will be operated on. The case was thoroughly discussed with the family. The possible risks and complications were explained and no guarantees were made.

OPERATIVE TECHNIQUE: With the patient under satisfactory general anesthesia and in the supine position, the lower abdomen and external genitalia were prepared and draped in the usual manner.
The operative site was infiltrated with 0.75% Marcaine solution. The right lower abdominal crease incision was made and was deepened down through the subcutaneous tissue.
After which, the external oblique aponeurosis was incised along its fibers down to the external ring. The cremaster muscle veil was carefully dissected free of the cord and at this time , it was noted at the level of the external ring. The testis was then carefully mobilized from its attack fibroareolar attachment and the cremaster muscle veil was carefully dissected all the way up to the internal ring. This maneuver allowed the testis to be brought down to the level of the scrotum without any tension. The tunica vaginalis was opened, and there was no evidence of any communicating inguinal hernia. However, there were an appendix testis and appendix epididymis, both of which were excised accordingly using short bursts of electrocautery.
The tunnel was then made from the groin wound towards the scrotum and the right hemiscrotal incision was made and a dartos pouch was created in between the scrotum and the dartos tunic. An opening was then made in the dartos tunic and a clamp was placed over the glove finger, which was then withdrawn into the operative inguinal field. After which, the testis was then carefully positioned and the inferior part of testicular structure was then carefully grasped with a clamp and the clamp was withdrawn through the tunnel and through the opening in the dartos tunic.
The testis was now fixated in the dartos pouch. The dartos fascia edges were made snug on either side of the incision with 4-0 chromic catgut stitch taking a bite of the tunica albuginea on either side of the testis. Once this was accomplished, the testis was carefully positioned into the dartos pouch without any tension and the scrotal incision edges were then approximated with interrupted
4-0 chromic catgut sutures. The groin wound was now checked for, there was no evidence of any tension or any twisting. Hemostasis was established. The wound was then closed in layers approximating the external oblique aponeurosis edges with interrupted 3-0 silk, Scarpa's fascia edges with interrupted 4-0 silk, Camper's fascia edges with interrupted 4-0 chromic catgut sutures, and the skin edges with a subcuticular 4-0 Monocryl suture. Steri- Strips were then applied over the wound, followed by Silvadene cream and a dry occlusive dressing. The child was then carefully awakened and taken out of the operating room in a satisfactory condition.

BLOOD LOSS: Minimal.

FLUID REPLACEMENT: Normal saline solution.

WOUND EXPECTANCY: Clean.'''

def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        #word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):

    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    # Saving model to disk
    pickle.dump(summary, open('summ.pkl','wb')) 


# Loading model to compare the results
    model = pickle.load(open('summ.pkl','rb'))

    return summary
    
     # 0 Reading the file content


if __name__ == '__main__':
    text_str = file_read('test.txt')
    #print(text_str)
    result = run_summarization(text_str)
    print("Summarised Result")
    print(result)

