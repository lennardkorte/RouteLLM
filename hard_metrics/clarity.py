import nltk
import time
from nltk.corpus import brown, reuters, gutenberg, webtext, abc, inaugural, nps_chat, udhr, movie_reviews, wordnet as wn
import textstat, spacy, pprint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

# Download the necessary corpora only once
nltk.download([
    'vader_lexicon', 'wordnet', 'omw-1.4', 'brown', 'reuters',
    'inaugural', 'gutenberg', 'webtext', 'abc', 'nps_chat', 'udhr', 'movie_reviews'
])

nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

prompt = """
In the realm of modern mathematics, one particularly intricate and fascinating area is the theory of \textit{moduli spaces}, which are geometric structures that classify certain types of mathematical objects up to equivalence. One of the most prominent and complex moduli spaces is the \textbf{Moduli Space of Riemann Surfaces}, often denoted as \( \mathcal{M}_g \), where \( g \) refers to the genus of the surface—essentially the number of "holes" or "handles" in the surface.

\section{The Moduli Space of Riemann Surfaces}

A Riemann surface is a one-dimensional complex manifold, meaning it locally behaves like the complex plane \( \mathbb{C} \), but globally can have a much richer and more complicated structure. The genus \( g \) of a Riemann surface plays a crucial role in determining the topological complexity of the surface. For example, a surface of genus 0 is the Riemann sphere, which is the simplest case, while genus 1 corresponds to a torus, and higher genus surfaces have increasingly intricate topologies.

The moduli space \( \mathcal{M}_g \) is the space that classifies all Riemann surfaces of a fixed genus \( g \), up to biholomorphic equivalence. In other words, it considers two Riemann surfaces to be the same if one can be smoothly transformed into the other via a complex-analytic map. The study of these spaces is extraordinarily rich and combines elements from algebraic geometry, differential geometry, and complex analysis.

\section{Teichmüller Space and Moduli Space}

The construction of the moduli space \( \mathcal{M}_g \) is closely related to another space called the \textit{Teichmüller space}, denoted \( \mathcal{T}_g \). While the moduli space \( \mathcal{M}_g \) captures the global structure of equivalence classes of Riemann surfaces, the Teichmüller space represents more detailed geometric information, distinguishing surfaces by additional structure such as marking of fundamental group homotopy classes.

Teichmüller space \( \mathcal{T}_g \) can be thought of as a universal cover of the moduli space \( \mathcal{M}_g \). The moduli space can be obtained from the Teichmüller space by taking the quotient by the action of a large discrete group, known as the \textit{Mapping Class Group}. This group acts on \( \mathcal{T}_g \) by changing the markings on the surfaces, and the quotient captures the idea that different markings may correspond to the same equivalence class of Riemann surfaces.

\section{Complex Dimension of the Moduli Space}

The dimension of the moduli space \( \mathcal{M}_g \) is a key feature. It is known that the complex dimension of \( \mathcal{M}_g \) is \( 3g - 3 \) for \( g \geq 2 \), a result that follows from the rich interplay between the geometry and topology of surfaces. For genus \( g = 1 \), the moduli space is 1-dimensional and is closely related to the \textit{upper half-plane model} of the modular group acting on lattices in \( \mathbb{C} \), forming a quotient space that classifies elliptic curves.

\section{Geometric Structures and the Weil-Petersson Metric}

One of the deep aspects of moduli space theory is the study of \textit{geometric structures} on \( \mathcal{M}_g \), particularly the Weil-Petersson metric. This is a natural Kähler metric on the moduli space, arising from the variation of the complex structure of Riemann surfaces. Unlike other metrics in differential geometry, the Weil-Petersson metric is incomplete, meaning that there are geodesics that "leave" the moduli space in finite time.

However, the singularities of this metric encode important geometric and topological information about the degeneration of Riemann surfaces. As a surface degenerates, it may approach the boundary of moduli space, where certain curves on the surface shrink to points, leading to a so-called \textit{nodal surface}.

\section{Open Problems in Moduli Space Theory}

Despite centuries of work, many deep problems remain unsolved in the theory of moduli spaces. For example:

\begin{itemize}
    \item \textbf{Understanding the global geometry of \( \mathcal{M}_g \)}: While local properties of the moduli space are well-understood, its global topological and geometric structure is much more elusive. For instance, finding a detailed description of the \textit{Chern classes} of the moduli space remains an active area of research.
    
    \item \textbf{Compactification of the Moduli Space}: The Deligne-Mumford compactification of \( \mathcal{M}_g \), which adds boundary components corresponding to degenerate surfaces, is a crucial tool for many purposes, but the detailed geometric behavior near the boundary is still not fully understood.
    
    \item \textbf{Higher Genus Moduli Spaces}: For higher genus surfaces, particularly for \( g \geq 6 \), the combinatorics of the degenerations and the structure of the boundary of the moduli space become extremely intricate, leading to deep challenges in understanding the full picture of the moduli space.
\end{itemize}

\section{Task: Investigating the Degeneration of Riemann Surfaces}

\textbf{Problem:} Consider a family of Riemann surfaces \( S_t \) parametrized by a complex variable \( t \), such that as \( t \to 0 \), the surface \( S_t \) undergoes a degeneration, resulting in a nodal surface when \( t = 0 \). Your task is to:

\begin{enumerate}
    \item \textbf{Describe the process of degeneration geometrically}: Specifically, explain how certain closed curves on the surface shrink to points and how this leads to the formation of nodes in the limiting surface.
    
    \item \textbf{Analyze the effect of the degeneration on the Weil-Petersson metric}: How does the metric behave as \( t \to 0 \)? What does this suggest about the structure of the moduli space near its boundary?
    
    \item \textbf{Optional extension}: Investigate the Deligne-Mumford compactification of the moduli space for genus 2. How does the boundary of the compactified moduli space relate to the degenerations you have described?
\end{enumerate}

This problem touches on the geometry and topology of moduli spaces, as well as the behavior of complex structures under degeneration—a central topic in modern algebraic geometry.
"""

# Readability Metrics: Returns floating-point readability scores
def readability_metrics(text):
    return {
        "Flesch-Kincaid Grade Level": textstat.flesch_kincaid_grade(text),
        "Gunning Fog Index": textstat.gunning_fog(text)
    }

# Sentence Complexity: Returns floating-point sentence complexity measures
def sentence_complexity(doc):
    sentences = list(doc.sents)
    total_words = sum(len(sentence) for sentence in sentences)
    total_word_length = sum(len(word) for word in doc if word.is_alpha)

    return {
        "Average Sentence Length": total_words / len(sentences) if sentences else 0,
        "Average Word Length": total_word_length / total_words if total_words else 0
    }

# Lexical Ambiguity: Calculates total synsets for each word and returns the average ambiguity score
def lexical_ambiguity(doc):
    total_synsets = sum(len(wn.synsets(token.text)) if len(wn.synsets(token.text)) > 0 else 1 for token in doc if token.is_alpha)
    total_words = sum(1 for token in doc if token.is_alpha)

    return {"Average Ambiguity Score": total_synsets / total_words if total_words else 0}

# Syntactic Complexity: Calculate average dependency length and number of clauses
def syntactic_complexity(doc):
    total_dependencies = 0
    total_clauses = 0

    for token in doc:
        if token.dep_ != "ROOT":
            total_dependencies += abs(token.head.i - token.i)
        if token.dep_ in ("csubj", "advcl", "relcl", "acl"):
            total_clauses += 1
    
    return {
        "Average Dependency Length": total_dependencies / len(doc) if len(doc) else 0,
        "Number of Clauses": total_clauses
    }

# Cohesion/Coherence: Calculate pairwise cosine similarity between sentence embeddings
def cohesion_coherence(doc):
    sentence_vectors = [sent.vector for sent in doc.sents if sent.has_vector]
    
    if len(sentence_vectors) < 2:
        return {"Average Sentence Cohesion": 1.0}

    similarity_matrix = cosine_similarity(sentence_vectors)
    n = len(sentence_vectors)
    avg_similarity = np.sum(np.triu(similarity_matrix, 1)) / (n * (n - 1) / 2) if n > 1 else 0

    return {"Average Sentence Cohesion": avg_similarity}

# Parse Tree Complexity: Calculates depth and branching factor of the parse tree
def parse_tree_complexity(doc):
    total_depth = 0
    total_branching_factors = []
    
    for sentence in doc.sents:
        sentence_depth = max([token.i - token.head.i for token in sentence if token.dep_ != "ROOT"], default=0)
        total_depth += sentence_depth
        
        for token in sentence:
            num_children = len(list(token.children))
            total_branching_factors.append(num_children)
    
    average_depth = total_depth / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0
    average_branching_factor = np.mean(total_branching_factors) if total_branching_factors else 0
    
    return {
        "Average Parse Tree Depth": average_depth,
        "Average Branching Factor": average_branching_factor
    }

# Vocabulary Diversity: Measures lexical diversity using type-token ratio
def vocabulary_diversity(text):
    words = text.split()
    unique_words = set(words)
    return {"Lexical Diversity": len(unique_words) / len(words) if words else 0}

# Contextuality: Calculate average similarity between individual sentences and the overall prompt
def contextuality(doc):
    overall_vector = doc.vector
    sentence_vectors = [sent.vector for sent in doc.sents if sent.has_vector]
    
    if len(sentence_vectors) < 2:
        return {"Contextuality": 1.0}

    similarities = [cosine_similarity([overall_vector], [sent_vector])[0][0] for sent_vector in sentence_vectors]
    avg_similarity = np.mean(similarities)
    
    return {"Contextuality": avg_similarity}

# Polarity and Subjectivity: Sentiment analysis of the prompt
def polarity_and_subjectivity(text):
    sentiment_scores = sid.polarity_scores(text)
    return {
        "Polarity": sentiment_scores['compound'],
        "Subjectivity": sentiment_scores['pos'] - sentiment_scores['neg']
    }

# Entropy of Word Choices: Shannon Entropy of word distribution in the prompt
def entropy_of_word_choices(text):
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)
    
    entropy = -sum((count / total_words) * np.log2(count / total_words) for count in word_counts.values())
    return {"Entropy of Word Choices": entropy}

# Polysemy Measure
def polysemy_measure(doc):
    polysemous_words = [token.text for token in doc if len(wn.synsets(token.text)) > 1]
    return {"Polysemous Words Proportion": len(polysemous_words) / len(doc) if len(doc) else 0}

# Intent Classification
def intent_classification(text):
    if text.endswith('?'):
        return {"Intent": "Question"}
    elif text.split()[0].lower() in ("generate", "explain", "list", "describe"):
        return {"Intent": "Instruction"}
    else:
        return {"Intent": "Statement"}

# Sentiment Balance
def sentiment_balance(text):
    sentiment_scores = sid.polarity_scores(text)
    total_sentiment = sentiment_scores['pos'] + sentiment_scores['neg'] + sentiment_scores['neu']
    return {
        "Positive Sentiment Proportion": sentiment_scores['pos'] / total_sentiment,
        "Negative Sentiment Proportion": sentiment_scores['neg'] / total_sentiment,
        "Neutral Sentiment Proportion": sentiment_scores['neu'] / total_sentiment
    }

# Redundancy Detection
def redundancy_detection(text):
    words = text.split()
    word_count = Counter(words)
    redundant_words = [word for word, count in word_count.items() if count > 1]
    return {"Redundancy Score": len(redundant_words) / len(words) if words else 0}

# Ambiguity Resolution
def ambiguity_resolution(doc):
    resolved_ambiguities = 0
    ambiguous_words = 0
    for token in doc:
        synsets = wn.synsets(token.text)
        if len(synsets) > 1:
            ambiguous_words += 1
            if token.has_vector:
                resolved_ambiguities += 1
    return {"Ambiguity Resolution": resolved_ambiguities / ambiguous_words if ambiguous_words else 0}

# Information Density
def information_density(doc):
    entities = [ent.text for ent in doc.ents]
    total_words = len([token for token in doc if token.is_alpha])
    return {"Information Density": len(entities) / total_words if total_words else 0}

# Topic Drift
def topic_drift(doc):
    sentences = list(doc.sents)
    sentence_vectors = [sent.vector for sent in sentences if sent.has_vector]
    if len(sentence_vectors) < 2:
        return {"Topic Drift": 0.0}
    
    similarity_matrix = cosine_similarity(sentence_vectors)
    avg_similarity = np.mean(np.triu(similarity_matrix, 1))

    return {"Topic Drift": 1.0 - avg_similarity}

# Question Complexity
def question_complexity(doc):
    questions = [sent for sent in doc.sents if sent.text.endswith('?')]
    complex_questions = 0
    for question in questions:
        num_clauses = sum(1 for token in question if token.dep_ in ("csubj", "advcl", "relcl", "acl"))
        if num_clauses > 1:
            complex_questions += 1
    return {"Complex Question Proportion": complex_questions / len(questions) if questions else 0}

# Contradiction Detection
def contradiction_detection(doc):
    contradictions = 0
    contradiction_indicators = ['but', 'however', 'yet', 'although', 'nevertheless', 'despite']
    
    for sent in doc.sents:
        if any(indicator in sent.text.lower() for indicator in contradiction_indicators):
            contradictions += 1
    return {"Contradiction Proportion": contradictions / len(list(doc.sents)) if doc.sents else 0}

# Specificity Measure
corpora_freq_cache = None
def domain_specific_frequency():
    global corpora_freq_cache
    if corpora_freq_cache is not None:
        return corpora_freq_cache

    brown_freq = set(brown.words())
    reuters_freq = set(reuters.words())
    gutenberg_freq = set(gutenberg.words())
    webtext_freq = set(webtext.words())
    abc_freq = set(abc.words())
    nps_chat_freq = set(nps_chat.words())
    inaugural_freq = set(inaugural.words())
    udhr_freq = set(udhr.words())
    movie_reviews_freq = set(movie_reviews.words())

    # Store the frequencies in a cache to avoid re-calculating
    corpora_freq_cache = {
        "General": brown_freq,
        "Economics/Business": reuters_freq,
        "Literature": gutenberg_freq,
        "Web/Informal": webtext_freq,
        "News": abc_freq,
        "Social Media": nps_chat_freq,
        "Religious/Political": inaugural_freq,
        "Human Rights": udhr_freq,
        "Sentiment/Movie Reviews": movie_reviews_freq,
    }
    
    return corpora_freq_cache

def specificity_measure(doc):
    corpora_freq = domain_specific_frequency()
    general_words = 0
    specific_words = 0
    domain_specific_words = {domain: 0 for domain in corpora_freq}

    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            if lemma in corpora_freq["General"]:
                general_words += 1
            else:
                specific_words += 1
                for domain, freq_dist in corpora_freq.items():
                    if lemma in freq_dist:
                        domain_specific_words[domain] += 1

    total_words = general_words + specific_words
    specificity_score = specific_words / total_words if total_words else 0
    domain_specificity_scores = {domain: count / total_words if total_words else 0 for domain, count in domain_specific_words.items()}

    return {
        "Specificity Score": specificity_score,
        **domain_specificity_scores
    }

# Task Complexity: Detects multiple actions or goals in the prompt
def task_complexity(text):
    action_words = [
        "explain", "describe", "list", "analyze", "summarize", "compare", "generate", "create", "define",
        "evaluate", "discuss", "outline", "examine", "assess", "interpret", "classify", "categorize",
        "justify", "identify", "demonstrate", "construct", "determine", "contrast", "review", "predict",
        "support", "critique", "argue", "explore", "illustrate", "develop", "design", "formulate",
        "solve", "organize", "prove", "synthesize", "exemplify", "debate", "rank"
    ]
    conjunctions = ["and", "or", ",", ";"]

    # Split text into words and lowercase them
    words = text.lower().split()

    # Count action words in the prompt
    action_word_count = sum(1 for word in words if word in action_words)

    # Count conjunctions and punctuation that may indicate multiple actions
    conjunction_count = sum(1 for word in words if word in conjunctions)

    # Estimate complexity based on action words and conjunctions
    complexity_score = action_word_count + conjunction_count

    # Return the complexity score and classification
    return {
        "Task Complexity Score": complexity_score,
    }

# Simple anaphora detection using rule-based heuristics
def anaphora_detection(doc):
    anaphora_pronouns = {
        'he', 'she', 'it', 'this', 'that', 'they', 'those', 'these',    
        'him', 'her', 'them', 'his', 'her', 'its', 'their', 'theirs',
        'himself', 'herself', 'itself', 'themselves', 'who', 'whom', 'whose', 'which', 'that',
        'one', 'ones', 'another', 'each', 'either', 'neither', 'anyone', 'someone', 'everyone',
        'none', 'both', 'all', 'any', 'some', 'few', 'other', 'others', 'such',
        'we', 'us', 'our', 'ours', 'myself', 'yourself', 'ourselves'
    }
    
    unresolved_anaphora = []
    sentences = list(doc.sents)
    
    for i, sent in enumerate(sentences):
        words = [token for token in sent if token.is_alpha]
        for j, word in enumerate(words):
            if word.text.lower() in anaphora_pronouns:
                antecedent_found = False
                
                for prev_word in words[:j]:
                    if prev_word.pos_ == 'NOUN':  
                        antecedent_found = True
                        break
                
                if not antecedent_found and i > 0:
                    previous_sentence_words = [token for token in sentences[i-1] if token.is_alpha]
                    for prev_word in previous_sentence_words:
                        if prev_word.pos_ == 'NOUN':  
                            antecedent_found = True
                            break
                
                if not antecedent_found:
                    unresolved_anaphora.append(word.text.lower())
    
    return {
        "Unresolved Anaphora Count": len(unresolved_anaphora),
    }

# Named Entity Repetition: Detect and count repetition of named entities
def named_entity_repetition(doc):
    entity_counts = Counter([ent.text for ent in doc.ents])
    total_entities = sum(entity_counts.values())
    repeated_entities = [entity for entity, count in entity_counts.items() if count > 1]
    
    # Calculate proportion of repeated entities
    repetition_score = len(repeated_entities) / total_entities if total_entities > 0 else 0
    
    return {
        "Named Entity Repetition Score": repetition_score,
    }

# Concreteness Measure using WordNet synsets
def concreteness_measure(doc):
    total_words = 0
    concrete_words = 0
    abstract_words = 0

    synset_cache = defaultdict(list)
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()

            if lemma in synset_cache:
                synsets = synset_cache[lemma]
            else:
                synsets = wn.synsets(lemma)
                synset_cache[lemma] = synsets  

            if synsets:
                if any(syn.pos() in ['n', 'v'] for syn in synsets):  
                    concrete_words += 1
                else:
                    abstract_words += 1
            else:
                abstract_words += 1

            total_words += 1

    concreteness_score = concrete_words / total_words if total_words > 0 else 0
    return {
        "Concreteness Score": concreteness_score,
        "Abstract Words Proportion": abstract_words / total_words if total_words > 0 else 0,
        "Concrete Words Proportion": concrete_words / total_words if total_words > 0 else 0,
    }

# Coreference Consistency: Checks if resolved coreference matches original entities
def coreference_consistency(doc):
    coref_resolved_doc = doc._.coref_resolved if doc.has_extension("coref_resolved") else None

    if not coref_resolved_doc:
        return {"Coreference Consistency": 1.0}
    
    pronoun_entities = [token for token in doc if token.pos_ == "PRON"]
    inconsistent_references = 0
    total_pronouns = len(pronoun_entities)

    for pronoun in pronoun_entities:
        resolved_entity = pronoun._.coref_cluster.main if pronoun._.in_coref else None
        if resolved_entity and resolved_entity != pronoun.head:
            inconsistent_references += 1

    consistency_score = 1.0 - (inconsistent_references / total_pronouns if total_pronouns > 0 else 0)

    return {"Coreference Consistency": consistency_score}

# Sentence Flow Fluency: Measures how smoothly sentences flow based on cosine similarity
def sentence_flow_fluency(doc):
    sentence_vectors = [sent.vector for sent in doc.sents if sent.has_vector]
    
    if len(sentence_vectors) < 2:
        return {"Sentence Flow Fluency": 1.0}
    
    similarities = []
    
    for i in range(len(sentence_vectors) - 1):
        similarity = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])[0][0]
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    return {"Sentence Flow Fluency": avg_similarity}

# Wrap each function call with timing
def timed_function_call(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Measure the time taken for each function and run prompt_properties with timing
def prompt_properties_with_timing(text):
    timings = {}
    start_time = time.time()
    doc, _ = timed_function_call(nlp, text)
    end_time = time.time()
    timings['nlp'] = end_time - start_time

    results = {}
    
    results['readability_metrics'], timings['readability_metrics'] = timed_function_call(readability_metrics, text)
    results['sentence_complexity'], timings['sentence_complexity'] = timed_function_call(sentence_complexity, doc)
    results['lexical_ambiguity'], timings['lexical_ambiguity'] = timed_function_call(lexical_ambiguity, doc)
    results['syntactic_complexity'], timings['syntactic_complexity'] = timed_function_call(syntactic_complexity, doc)
    results['parse_tree_complexity'], timings['parse_tree_complexity'] = timed_function_call(parse_tree_complexity, doc)
    results['cohesion_coherence'], timings['cohesion_coherence'] = timed_function_call(cohesion_coherence, doc)
    results['vocabulary_diversity'], timings['vocabulary_diversity'] = timed_function_call(vocabulary_diversity, text)
    results['contextuality'], timings['contextuality'] = timed_function_call(contextuality, doc)
    results['polarity_and_subjectivity'], timings['polarity_and_subjectivity'] = timed_function_call(polarity_and_subjectivity, text)
    results['entropy_of_word_choices'], timings['entropy_of_word_choices'] = timed_function_call(entropy_of_word_choices, text)
    results['polysemy_measure'], timings['polysemy_measure'] = timed_function_call(polysemy_measure, doc)
    results['intent_classification'], timings['intent_classification'] = timed_function_call(intent_classification, text)
    results['sentiment_balance'], timings['sentiment_balance'] = timed_function_call(sentiment_balance, text)
    results['redundancy_detection'], timings['redundancy_detection'] = timed_function_call(redundancy_detection, text)
    results['information_density'], timings['information_density'] = timed_function_call(information_density, doc)
    results['ambiguity_resolution'], timings['ambiguity_resolution'] = timed_function_call(ambiguity_resolution, doc)
    results['topic_drift'], timings['topic_drift'] = timed_function_call(topic_drift, doc)
    results['question_complexity'], timings['question_complexity'] = timed_function_call(question_complexity, doc)
    results['specificity_measure'], timings['specificity_measure'] = timed_function_call(specificity_measure, doc)
    results['contradiction_detection'], timings['contradiction_detection'] = timed_function_call(contradiction_detection, doc)
    results['task_complexity'], timings['task_complexity'] = timed_function_call(task_complexity, text)
    results['anaphora_detection'], timings['anaphora_detection'] = timed_function_call(anaphora_detection, doc)
    results['named_entity_repetition'], timings['named_entity_repetition'] = timed_function_call(named_entity_repetition, doc)
    results['concreteness_measure'], timings['concreteness_measure'] = timed_function_call(concreteness_measure, doc)
    results['coreference_consistency'], timings['coreference_consistency'] = timed_function_call(coreference_consistency, doc)
    results['sentence_flow_fluency'], timings['sentence_flow_fluency'] = timed_function_call(sentence_flow_fluency, doc)

    return results, timings

results, timings = prompt_properties_with_timing(prompt)
results2, timings2 = prompt_properties_with_timing(prompt)
pprint.pprint(results2)

def print_timings(timings):
    print("Execution Times for Each Function:")
    for function_name, time_taken in timings.items():
        print(f"{function_name}: {time_taken:.6f} seconds")

print_timings(timings2)

def compute_total_execution_time(timings):
    return sum(timings.values())

# Print the total execution time for 'timings2'
total_execution_time = compute_total_execution_time(timings2)
print(f"Total Execution Time: {total_execution_time:.6f} seconds")

# Print the updated clarity metrics

