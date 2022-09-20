from src.constants import *
from src.models import DeclModel
import re


def modify_condition(condition):
    if condition.strip() == "":
        condition = "True"
        return condition
    if "is" in condition:
        condition = condition.replace("is", "==")
    words = condition.split()
    for index, word in enumerate(words):
        if "A." in word:
            words[index] = "A[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif "T." in word:
            words[index] = "T[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif word == "same":
            words[index] = "A[\"" + words[index + 1] + "\"] == T[\"" + words[index + 1] + "\"]"
            words[index + 1] = ""
    words = list(filter(lambda word: word != "", words))
    condition = " ".join(words)
    return condition


def parse_decl(path):
    fo = open(path, "r+")
    lines = fo.readlines()
    result = DeclModel()
    result.checkers = list()
    for line in lines:
        if line.startswith('activity'):
            result.activities.append(line.split()[1])
        elif (line.startswith(EXISTENCE)
            or line.startswith(ABSENCE)
            or line.startswith(EXACTLY)):
            split = line.split("[")
            key = split[0].strip()
            attribute = split[1].split("]")[0]
            n = 1
            if any(map(str.isdigit, key)):
                n = int(re.search(r'\d+', key).group())
            tmp = dict()
            tmp['key'] = key
            tmp['condition'] = [modify_condition(line.split("|")[1]), n]
            tmp['attribute'] = attribute
            result.checkers.append(tmp)
        elif (line.startswith(INIT)
            or line.startswith(CHOICE)
            or line.startswith(EXCLUSIVE_CHOICE)):
            split = line.split("[")
            key = split[0].strip()
            attribute = split[1].split("]")[0]
            n = 1
            if any(map(str.isdigit, key)):
                n = int(re.search(r'\d+', key).group())
            tmp = dict()
            tmp['key'] = key
            tmp['condition'] = [modify_condition(line.split("|")[1]), n]
            tmp['attribute'] = attribute
            result.checkers.append(tmp)
        elif (line.startswith(RESPONDED_EXISTENCE)
            or line.startswith(RESPONSE)
            or line.startswith(ALTERNATE_RESPONSE)
            or line.startswith(CHAIN_RESPONSE)
            or line.startswith(PRECEDENCE)
            or line.startswith(ALTERNATE_PRECEDENCE)
            or line.startswith(CHAIN_PRECEDENCE)
            or line.startswith(NOT_RESPONDED_EXISTENCE)
            or line.startswith(NOT_RESPONSE)
            or line.startswith(NOT_CHAIN_RESPONSE)
            or line.startswith(NOT_PRECEDENCE)
            or line.startswith(NOT_CHAIN_PRECEDENCE)):
            split = line.split("[")
            key = split[0].strip()
            attribute = split[1].split("]")[0]
            tmp = dict()
            tmp['key'] = key
            tmp['condition'] = [modify_condition(line.split("|")[1]), modify_condition(line.split("|")[2])]
            tmp['attribute'] = attribute
            result.checkers.append(tmp)
        elif (line.startswith(CHAIN_SUCCESSION)
            or line.startswith(CHAIN_SUCCESSION)
            or line.startswith(SUCCESSION)):
            split = line.split("[")
            key = split[0].strip()
            attribute = split[1].split("]")[0]
            tmp = dict()
            tmp['key'] = key
            tmp['condition'] = [modify_condition(line.split("|")[1]), modify_condition(line.split("|")[2])]
            tmp['attribute'] = attribute
    fo.close()
    return result
