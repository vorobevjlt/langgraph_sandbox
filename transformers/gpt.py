with open('/Users/justlikethat/langgraph-course/resourses/artical_web_rag.txt','r') as file:
    text = " ".join(line.rstrip() for line in file)

a = text[:10]
print(sorted(list(set(a))))
