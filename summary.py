import os, re
import networkx as nx
from rouge import Rouge

'''
Sisaldab koodi siit https://github.com/soras/EstTimeMLCorpus/blob/master/exported_corpus_reader.py
Vajalik korpus kättesaadav: https://github.com/soras/EstTimeMLCorpus
'''

baseAnnotationFile = "base-segmentation-morph-syntax"
eventAnnotationFile = "event-annotation"
timexAnnotationFile = "timex-annotation"
timexAnnotationDCTFile = "timex-annotation-dct"
tlinkEventTimexFile = "tlink-event-timex"
tlinkEventDCTFile = "tlink-event-dct"
tlinkMainEventsFile = "tlink-main-events"
tlinkSubEventsFile = "tlink-subordinate-events"


# =========================================================================
#    Loading corpus files
# =========================================================================

def load_base_segmentation(inputFile):
    base_segmentation = dict()
    last_sentenceID = ""
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if (re.match("^#.+$", line)):
            continue
        items = (line.rstrip()).split("\t")
        if (len(items) != 7):
            raise Exception(" Unexpected number of items on line: '" + str(line) + "'")
        file = items[0]
        if (file not in base_segmentation):
            base_segmentation[file] = []
        sentenceID = items[1]
        if (sentenceID != last_sentenceID):
            base_segmentation[file].append([])
        wordID = items[2]
        # fileName	sentence_ID	word_ID_in_sentence	token	morphological_and_syntactic_annotations	syntactic_ID	syntactic_ID_of_head
        token = items[3]
        morphSyntactic = items[4]
        syntacticID = items[5]
        syntacticHeadID = items[6]
        base_segmentation[file][-1].append([sentenceID, wordID, token, morphSyntactic, syntacticID, syntacticHeadID])
        last_sentenceID = sentenceID
    f.close()
    return base_segmentation


def load_entity_annotation(inputFile):
    annotationsByLoc = dict()
    annotationsByID = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if (re.match("^#.+$", line)):
            continue
        items = (line.rstrip()).split("\t")
        # fileName	sentence_ID	word_ID_in_sentence	expression	event_annotation	event_ID
        # fileName	sentence_ID	word_ID_in_sentence	expression	timex_annotation	timex_ID
        if (len(items) != 6):
            raise Exception(" Unexpected number of items on line: '" + str(line) + "'")
        file = items[0]
        sentenceID = items[1]
        wordID = items[2]
        expression = items[3]
        annotation = items[4]
        entityID = items[5]
        if (file not in annotationsByLoc):
            annotationsByLoc[file] = dict()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        # Record annotation by its location in text
        locKey = (sentenceID, wordID)
        if (locKey not in annotationsByLoc[file]):
            annotationsByLoc[file][locKey] = []
        annotationsByLoc[file][locKey].append([entityID, expression, annotation])
        # Record annotation by its unique ID in text
        if (entityID not in annotationsByID[file]):
            annotationsByID[file][entityID] = []
        annotationsByID[file][entityID].append([sentenceID, wordID, expression, annotation])
    f.close()
    return (annotationsByLoc, annotationsByID)


def load_dct_annotation(inputFile):
    DCTsByFile = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if (re.match("^#.+$", line)):
            continue
        items = (line.rstrip()).split("\t")
        # fileName	document_creation_time
        if (len(items) != 2):
            raise Exception(" Unexpected number of items on line: '" + str(line) + "'")
        file = items[0]
        dct = items[1]
        DCTsByFile[file] = dct
    f.close()
    return DCTsByFile


def load_relation_annotation(inputFile):
    annotationsByID = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if (re.match("^#.+$", line)):
            continue
        items = line.split("\t")
        # old format: fileName	entityID_A	relation	entityID_B	comment	expression_A	expression_B
        # new format: fileName	entityID_A	relation	entityID_B	comment
        if (len(items) != 5):
            print(len(items))
            raise Exception(" Unexpected number of items on line: '" + str(line) + "'")
        file = items[0]
        entityA = items[1]
        relation = items[2]
        entityB = items[3]
        comment = items[4].rstrip()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        annotation = [entityA, relation, entityB, comment]
        if (entityA not in annotationsByID[file]):
            annotationsByID[file][entityA] = []
        annotationsByID[file][entityA].append(annotation)
        if (entityB not in annotationsByID[file]):
            annotationsByID[file][entityB] = []
        annotationsByID[file][entityB].append(annotation)
    f.close()

    return annotationsByID


def load_relation_to_dct_annotations(inputFile):
    annotationsByID = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if (re.match("^#.+$", line)):
            continue
        items = line.split("\t")
        # old format: fileName	entityID_A	relation_to_DCT	comment	expression_A
        # new format: fileName	entityID_A	relation_to_DCT	comment
        if (len(items) != 4):
            raise Exception(" Unexpected number of items on line: '" + str(line) + "'")
        file = items[0]
        entityA = items[1]
        relationToDCT = items[2]
        comment = items[3].rstrip()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        annotation = [entityA, relationToDCT, "t0", comment]
        if (entityA not in annotationsByID[file]):
            annotationsByID[file][entityA] = []
        annotationsByID[file][entityA].append(annotation)
    f.close()

    return annotationsByID


# =========================================================================
#    Displaying annotations on corpus files
# =========================================================================
def getEntityIDsOfTheSentence(file, sentID, base, eventsByLoc, timexesByLoc):
    events = []
    timexes = []
    seenIDs = dict()
    for wordID in range(len(base[file][sentID])):
        [sID, wID, token, morphSyntactic, syntacticID, syntacticHeadID] = base[file][sentID][wordID]
        key = (sID, wID)
        if (file in eventsByLoc and key in eventsByLoc[file]):
            for [entityID, expression, annotation] in eventsByLoc[file][key]:
                if (entityID not in seenIDs):
                    events.append(entityID)
                    seenIDs[entityID] = 1
        if (file in timexesByLoc and key in timexesByLoc[file]):
            for [entityID, expression, annotation] in timexesByLoc[file][key]:
                if (entityID not in seenIDs):
                    timexes.append(entityID)
                    seenIDs[entityID] = 1
    return (events, timexes)


# Retrieves an expression corresponding to the entity
def getExpr(file, entityID, entitiesByIDs):
    if (entityID in entitiesByIDs[file]):
        # Collect entity expressions
        expressions = set()
        for item in entitiesByIDs[file][entityID]:
            # [sentenceID, wordID, expression, annotation]
            expressions.add(item[2])
        if (len(expressions) == 1):
            return expressions.pop()
        else:
            raise Exception(" Unexpected number of expressions for " + entityID + ": " + str(expressions))
    else:
        raise Exception(" Unable to the retrieve expression for the entity " + entityID)


def getTLINKAnnotations(file, eventIDs, eventsByID, mainEventLinks, subEventLinks):
    linkAnnotations = []
    for eventID in eventIDs:
        if file in subEventLinks and eventID in subEventLinks[file]:
            for annotation in subEventLinks[file][eventID]:
                [entityA, relation, entityB, comment] = annotation
                if eventID == entityA:
                    exprA = getExpr(file, entityA, eventsByID)
                    exprB = getExpr(file, entityB, eventsByID)
                    linkAnnotations.append(
                        {"entityA": entityA, "wordA": exprA, "relation": relation, "entityB": entityB, "wordB": exprB}
                    )
        if file in mainEventLinks and eventID in mainEventLinks[file]:
            for annotation in mainEventLinks[file][eventID]:
                [entityA, relation, entityB, comment] = annotation
                if eventID == entityA:
                    exprA = getExpr(file, entityA, eventsByID)
                    exprB = getExpr(file, entityB, eventsByID)
                    linkAnnotations.append(
                        {"entityA": entityA, "wordA": exprA, "relation": relation, "entityB": entityB, "wordB": exprB}
                    )
    return (linkAnnotations)


def links(base, eventsByLoc, timexesByLoc, eventsByID, timexesByID, DCTsByFile, eventTimexLinks, eventDCTLinks,
          mainEventLinks, subEventLinks, findSummaryMethod):
    summaries = {}
    """
    summaries{file:
                    {fulltext:full text,
                    summary:summary text.
                    summaryid:summary sentence id's}
    }
    
    """
    for file in sorted(base):
        fulltext = []
        values = {}
        text_links = []
        for sentID in range(len(base[file])):
            (eventIDs, timexIDs) = getEntityIDsOfTheSentence(file, sentID, base, eventsByLoc, timexesByLoc)
            linkAnnotations = getTLINKAnnotations(file, eventIDs, eventsByID, mainEventLinks, subEventLinks)
            text_links.extend(linkAnnotations)
            sentAnnotation = getSentenceWithEntityAnnotations(file, sentID, base, eventsByLoc, timexesByLoc)
            try:
                fulltext.append(sentAnnotation.strip())
            except:
                fulltext.append(sentAnnotation.encode("utf-8").strip())
        # Find summary
        sentences = findSummaryMethod(text_links, file)

        sentences.sort(key=lambda x: int(x[0]))
        sentences.sort(key=lambda x: float(x[1]), reverse=True)
        while (True):
            summary = []
            for sent in sentences:
                sentAnnotation = getSentenceWithEntityAnnotations(file, int(sent[0]), base, eventsByLoc, timexesByLoc)
                try:
                    summary.append(sentAnnotation.strip())
                except:
                    summary.append(sentAnnotation.encode("utf-8").strip())

            if (getSummaryProcentage(fulltext, summary) > 40 and len(sentences) > 1):
                sentences.pop()

            else:
                break
        sentences.sort(key=lambda x: int(x[0]))

        # Sentences in summary
        summary = []
        for sent in sentences:
            sentAnnotation = getSentenceWithEntityAnnotations(file, int(sent[0]), base, eventsByLoc, timexesByLoc)
            try:
                summary.append(sentAnnotation.strip())
            except:
                summary.append(sentAnnotation.encode("utf-8").strip())
        values["fulltext"] = fulltext
        values["summary"] = summary
        values["summaryid"] = sentences
        summaries[file] = values

    return summaries


def removeGraphCycles(Graph):
    '''
    Eemaldab graafist tsüklid
    '''
    while True:
        try:
            cycles = list(nx.find_cycle(Graph))
            max_in_degree = 0
            edge_to_remove = None
            for edge in cycles:
                if Graph.in_degree[edge[1]] > max_in_degree:
                    max_in_degree = Graph.in_degree[edge[1]]
                    edge_to_remove = edge

            Graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
        except nx.exception.NetworkXNoCycle:
            break

    return Graph


def all_paths(G):
    '''
    Leiab ettenatud graafist pikima ahela, tagastab kõik, kui neid on mitu
    '''
    all_paths = []
    all_longest_paths = []
    for node in G.nodes:
        for node2 in G.nodes:
            if node != node2:
                for path in nx.all_simple_paths(G, source=node, target=node2):
                    all_paths.append(path)

    max_len = 0
    for elem in all_paths:
        if nx.path_weight(G, elem, weight="weight") > max_len:
            max_len = nx.path_weight(G, elem, weight="weight")

    for path in all_paths:
        if nx.path_weight(G, path, weight="weight") == max_len:
            all_longest_paths.append(path)

    return all_longest_paths


def getSummaryProcentage(fulltext, summarytext):
    '''
        Kui palju lühem on kokkuvõte algtekstist
    '''
    fulltext = "".join(fulltext).replace(" ", "")
    summarytext = "".join(summarytext).replace(" ", "")
    fulltext_len = len(fulltext)
    summarytext_len = len(summarytext)
    return (100 * summarytext_len) / fulltext_len


def generateGraphMethod1(text_links, file):
    Graph = nx.DiGraph()
    nodes = []
    for relation in text_links:
        if relation["relation"] == "BEFORE":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="BEFORE")
        elif relation["relation"] == "AFTER":
            Graph.add_edge(relation["entityB"], relation["entityA"], weight=1, label="AFTER")
        else:
            continue
            # print("Else:",relation["relation"])
        nodes.append(relation["entityA"])
        nodes.append(relation["entityB"])

    Original_Graph = Graph.copy()
    nodes = set(nodes)
    largest_cc = max(nx.weakly_connected_components(Graph), key=len)
    for node in nodes:
        if node not in largest_cc:
            Graph.remove_node(node)

    removeGraphCycles(Graph)
    return Graph, Original_Graph


def generateGraphMethod2(text_links, file):
    Graph = nx.DiGraph()
    nodes = []
    for relation in text_links:
        if relation["relation"] == "BEFORE":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="BEFORE")
        elif relation["relation"] == "BEFORE-OR-OVERLAP":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=0.4, label="BEFORE-OR-OVERLAP")
        elif relation["relation"] == "AFTER":
            Graph.add_edge(relation["entityB"], relation["entityA"], weight=1, label="AFTER")
        elif relation["relation"] == "OVERLAP-OR-AFTER":
            Graph.add_edge(relation["entityB"], relation["entityA"], weight=0.4, label="OVERLAP-OR-AFTER")
        else:
            continue
        nodes.append(relation["entityA"])
        nodes.append(relation["entityB"])

    Original_Graph = Graph.copy()
    nodes = set(nodes)
    largest_cc = max(nx.weakly_connected_components(Graph), key=len)
    for node in nodes:
        if node not in largest_cc:
            Graph.remove_node(node)

    removeGraphCycles(Graph)
    return Graph, Original_Graph


def generateGraphMethod3(text_links, file):
    Graph = nx.DiGraph()
    nodes = []
    for relation in text_links:
        if relation["relation"] == "BEFORE":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="BEFORE")
        elif relation["relation"] == "SIMULTANEOUS":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="SIMULTANEOUS")
        elif relation["relation"] == "IS_INCLUDED":
            Graph.add_edge(relation["entityB"], relation["entityA"], weight=1, label="IS_INCLUDED")
        # elif relation["relation"] == "BEFORE-OR-OVERLAP":
        # Graph.add_edge(relation["entityA"], relation["entityB"], weight=0.4, label="BEFORE-OR-OVERLAP")
        elif relation["relation"] == "IDENTITY":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="IDENTITY")


        elif relation["relation"] == "AFTER":
            Graph.add_edge(relation["entityB"], relation["entityA"], weight=1, label="AFTER")
        elif relation["relation"] == "INCLUDES":
            Graph.add_edge(relation["entityA"], relation["entityB"], weight=1, label="INCLUDES")
        # elif relation["relation"] == "OVERLAP-OR-AFTER":
        # Graph.add_edge(relation["entityB"], relation["entityA"], weight=0.4, label="OVERLAP-OR-AFTER")
        else:
            continue
        nodes.append(relation["entityA"])
        nodes.append(relation["entityB"])

    Original_Graph = Graph.copy()
    nodes = set(nodes)
    largest_cc = max(nx.weakly_connected_components(Graph), key=len)
    for node in nodes:
        if node not in largest_cc:
            Graph.remove_node(node)

    removeGraphCycles(Graph)
    return Graph, Original_Graph


def findSummaryFromGraph_TermSignifiganceDegrees(Graph, file, graph_path, original_graph):
    '''
        Hindab etteantud ahelas olevate lausete tähtsust aste järgi algses graafis
    '''
    sentences = []
    path = []

    sum_of_all_degrees = 0
    for elem in graph_path:
        word = eventsByID[file][elem][0][2]
        sentence_id = eventsByID[file][elem][0][0]
        sentences.append(sentence_id)
        degree = original_graph.degree[elem]
        sum_of_all_degrees += degree
        d = {"wordID": elem, "word": word, "sentID": sentence_id, "degree": degree}
        path.append(d)

    sentences = set(sentences)
    # Sentence significance based on degrees
    sentence_sig = []
    for sent in sentences:
        sig = 0
        for elem in path:
            if elem["sentID"] == sent:
                sig += elem["degree"]
        sig = sig / sum_of_all_degrees
        sentence_sig.append((sent, sig))

    sentence_sig.sort(key=lambda x: x[1])
    sentence_sig.reverse()
    return sentence_sig


def GetSummaryFromGraphMethod1(text_links, file):
    '''
        Pikim ahel (üks pikim ahel) koos seostega BEFORE ja AFTER

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod1(text_links, file)
    longest_path = nx.dag_longest_path(g)
    return findSummaryFromGraph_TermSignifiganceDegrees(g, file, longest_path, original_g)


def GetSummaryFromGraphMethod2(text_links, file):
    '''
        PageRank koos BEFORE ja AFTER seostega

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod1(text_links, file)
    pagerank = nx.pagerank(original_g)
    sentences_dict = {}
    for id in pagerank:
        word_score = pagerank[id]
        sentence_id = eventsByID[file][id][0][0]
        if sentence_id in sentences_dict.keys():
            sentences_dict[sentence_id] = sentences_dict[sentence_id] + word_score
        else:
            sentences_dict[sentence_id] = word_score

    sentence_sig = []
    for elem in sentences_dict:
        sentence_sig.append((elem, sentences_dict[elem]))

    sentence_sig.sort(key=lambda x: x[1])
    sentence_sig.reverse()
    return sentence_sig


def GetSummaryFromGraphMethod3(text_links, file):
    '''
        Pikim ahel (võtakse kõik pikimad ahelaid, kui neid on mitu) koos seostega BEFORE ja AFTER

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod1(text_links, file)
    longest_paths = all_paths(g)
    longest_path = []
    for elem in longest_paths:
        longest_path.extend(elem)
    return findSummaryFromGraph_TermSignifiganceDegrees(g, file, longest_path, original_g)


def GetSummaryFromGraphMethod4(text_links, file):
    '''
        Pikim ahel (üks pikim ahel) koos seostega BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod2(text_links, file)
    longest_path = nx.dag_longest_path(g)
    return findSummaryFromGraph_TermSignifiganceDegrees(g, file, longest_path, original_g)


def GetSummaryFromGraphMethod5(text_links, file):
    '''
        PageRank koos seostega BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod2(text_links, file)
    pagerank = nx.pagerank(original_g)
    sentences_dict = {}
    for id in pagerank:
        word_score = pagerank[id]
        sentence_id = eventsByID[file][id][0][0]
        if sentence_id in sentences_dict.keys():
            sentences_dict[sentence_id] = sentences_dict[sentence_id] + word_score
        else:
            sentences_dict[sentence_id] = word_score

    sentence_sig = []
    for elem in sentences_dict:
        sentence_sig.append((elem, sentences_dict[elem]))

    sentence_sig.sort(key=lambda x: x[1])
    sentence_sig.reverse()
    return sentence_sig


def GetSummaryFromGraphMethod6(text_links, file):
    '''
        Pikim ahel (võtakse kõik pikimad ahelaid, kui neid on mitu) koos seostega BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod2(text_links, file)
    longest_paths = all_paths(g)
    longest_path = []
    for elem in longest_paths:
        longest_path.extend(elem)

    return findSummaryFromGraph_TermSignifiganceDegrees(g, file, longest_path, original_g)


def GetSummaryFromGraphMethod7(text_links, file):
    '''
        Pikim ahel (võtakse kõik pikimad ahelaid, kui neid on mitu) koos seostega BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod3(text_links, file)
    longest_paths = all_paths(g)
    longest_path = []
    for elem in longest_paths:
        longest_path.extend(elem)

    return findSummaryFromGraph_TermSignifiganceDegrees(g, file, longest_path, original_g)


def GetSummaryFromGraphMethod8(text_links, file):
    '''
        PageRank koos seostega BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod3(text_links, file)
    pagerank = nx.pagerank(original_g)
    sentences_dict = {}
    for id in pagerank:
        word_score = pagerank[id]
        sentence_id = eventsByID[file][id][0][0]
        if sentence_id in sentences_dict.keys():
            sentences_dict[sentence_id] = sentences_dict[sentence_id] + word_score
        else:
            sentences_dict[sentence_id] = word_score

    sentence_sig = []
    for elem in sentences_dict:
        sentence_sig.append((elem, sentences_dict[elem]))

    sentence_sig.sort(key=lambda x: x[1])
    sentence_sig.reverse()
    return sentence_sig


def GetSummaryFromGraphMethod9(text_links, file):
    '''
        Pikim ahel(kõik pikimad ahelad, ajaväljendite olemasolu) koos seostega BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY

        Tagastab laused koos nende tähtsus hinnanguga
    '''
    g, original_g = generateGraphMethod3(text_links, file)
    longest_paths = all_paths(g)
    longest_path = []
    for elem in longest_paths:
        longest_path.extend(elem)

    return SentenceSignificance(g, file, longest_path, original_g)


def SentenceSignificance(Graph, file, graph_path, original_graph):
    sentences = []
    path = []

    sum_of_all_degrees = 0
    for elem in graph_path:
        word = eventsByID[file][elem][0][2]
        sentence_id = eventsByID[file][elem][0][0]
        sentences.append(sentence_id)
        degree = original_graph.degree[elem]
        sum_of_all_degrees += degree
        d = {"wordID": elem, "word": word, "sentID": sentence_id, "degree": degree}
        path.append(d)

    sentences = set(sentences)
    # Sentece signifigance based on degrees
    sentence_sig = []
    for sent in sentences:
        sig = 0
        for elem in path:
            if elem["sentID"] == sent:
                sig += elem["degree"]
        sig = sig / sum_of_all_degrees

        try:
            for key in timexesByID[file].keys():
                if sent == timexesByID[file][key][0][0]:
                    sig += 0.4
                    break
        except:
            None
        sentence_sig.append((sent, sig))

    sentence_sig.sort(key=lambda x: x[1])
    sentence_sig.reverse()
    return sentence_sig


def get_all_summaries(summary_file_path):
    with open(summary_file_path, "r", encoding="utf8") as f:
        all_summaries = {}
        text = f.read()
        text = text.split("\n")
        for row in text:
            row = row.split("\t")
            all_summaries[row[0]] = row[1].split(",")
    return all_summaries


def evaluate(hy_summaries, ref_summaries):
    sum_len = 0
    arvud = []
    rouge = Rouge()
    scores = {}
    for artikkel in hy_summaries:
        hy = " ".join(hy_summaries[artikkel]["summary"]).strip()
        ref = []
        for elem in ref_summaries[artikkel]:
            ref.append(
                getSentenceWithEntityAnnotations(artikkel, int(elem), baseAnnotations, eventsByLoc,
                                                 timexesByLoc).strip())

        arvud.append(getSummaryProcentage(hy_summaries[artikkel]["fulltext"], ref))

        sum_len += getSummaryProcentage(hy_summaries[artikkel]["fulltext"], hy_summaries[artikkel]["summary"])
        ref = " ".join(ref)
        scores[artikkel] = rouge.get_scores(hy, ref)[0]["rouge-l"]

    # print("Käsitsi loodud kokkuvõtted on keskmiselt", round(sum(arvud) / len(arvud),2), "% lühemad")
    # print("Automaatselt loodud kokkuvütted on keskmiselt", round(sum_len / len(hy_summaries),2), "% lühemad")
    # print()

    return scores


def trainingTestScores(scores, training_set_path, test_set_path):
    with open(training_set_path, "r") as f:
        training_set = f.read().split("\n")
    with open(test_set_path, "r") as f:
        test_set = f.read().split("\n")

    training_set_scores = {}
    test_set_scores = {}
    for article in scores:
        if article in training_set:
            training_set_scores[article] = scores[article]
        elif article in test_set:
            test_set_scores[article] = scores[article]
        else:
            print("Article not in training or test set")

    return training_set_scores, test_set_scores


def printEvalScore(scores, print_all=False):
    recall_avg = 0
    precsion_avg = 0
    f_avg = 0

    for x in scores:
        if print_all:
            print(x, scores[x]["r"], "|", scores[x]["p"], "|", scores[x]["f"])
        recall_avg += scores[x]["r"]
        precsion_avg += scores[x]["p"]
        f_avg += scores[x]["f"]

    print("Artikleid:", len(scores))
    print("Saagis (Recall):", recall_avg / len(scores))
    print("Täpsus (Precision):", precsion_avg / len(scores))
    print("F-mõõt (F-measure):", f_avg / len(scores))


# /////////////////////////////////////////////////////////////////////

corpusDir = "corpus"

# Load base segmentation, morphological and syntactic annotations
baseSegmentationFile = os.path.join(corpusDir, baseAnnotationFile)
baseAnnotations = load_base_segmentation(baseSegmentationFile)

# Load EVENT, TIMEX annotations
(eventsByLoc, eventsByID) = load_entity_annotation(os.path.join(corpusDir, eventAnnotationFile))
(timexesByLoc, timexesByID) = load_entity_annotation(os.path.join(corpusDir, timexAnnotationFile))
DCTsByFile = load_dct_annotation(os.path.join(corpusDir, timexAnnotationDCTFile))

# Load TLINK annotations
eventTimexLinks = load_relation_annotation(os.path.join(corpusDir, tlinkEventTimexFile))
eventDCTLinks = load_relation_to_dct_annotations(os.path.join(corpusDir, tlinkEventDCTFile))
mainEventLinks = load_relation_annotation(os.path.join(corpusDir, tlinkMainEventsFile))
subEventLinks = load_relation_annotation(os.path.join(corpusDir, tlinkSubEventsFile))


def getSentenceWithEntityAnnotations(file, sentID, base, eventsByLoc, timexesByLoc):
    sentAnnotation = ""
    for wordID in range(len(base[file][sentID])):
        [sID, wID, token, morphSyntactic, syntacticID, syntacticHeadID] = base[file][sentID][wordID]
        key = (sID, wID)
        sentAnnotation += " " + token
    return sentAnnotation


methods = [GetSummaryFromGraphMethod1, GetSummaryFromGraphMethod2, GetSummaryFromGraphMethod3,
           GetSummaryFromGraphMethod4, GetSummaryFromGraphMethod5, GetSummaryFromGraphMethod6,
           GetSummaryFromGraphMethod7,
           GetSummaryFromGraphMethod8, GetSummaryFromGraphMethod9]
all_summaries = get_all_summaries("Kokkuvõtted.txt")

for method in methods:
    print(method.__name__)
    summaries = links(baseAnnotations, eventsByLoc, timexesByLoc, eventsByID, timexesByID, DCTsByFile, eventTimexLinks,
                      eventDCTLinks, mainEventLinks, subEventLinks, method)
    scores = evaluate(summaries, all_summaries)
    training_scores, test_scores = trainingTestScores(scores, "train_set.txt", "test_set.txt")
    # printEvalScore(scores,True)
    print("Arenduskorpus: ")
    printEvalScore(training_scores)
    print()
    print("Testkorpus")
    printEvalScore(test_scores)
    print("=" * 50)
