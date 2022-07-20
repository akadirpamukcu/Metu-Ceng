def bol(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def control(k,corpus):
	hs = len(corpus[0]) #harf sayisi
	ks = len(corpus) #verilen kelime sayisi
	bs = k.count("-") #bos satir sayisi
	p = len(k)-bs #dolu satir sayisi
	l = []
	sutuns = []
	sss = []
	a = 0
	if p == 0:
		return False
	for y in k:
		if y != "-":
			l.append(y)

	for n in range(hs):
		for s in l:
			sss.append(s[n])
	
	sssn =  list(bol(sss, p))
	for f in sssn:
		kk = "".join(f)
		sutuns.append(kk)

	for i in range (len(sutuns)-1):
		if sutuns[i] == sutuns[i+1]:
			return False

	for i in range (len(k)-1):
		if k[i] == k[i+1]:
			return False

	print "sutuns:" , sutuns
	for t in corpus:
		if t[:p] in sutuns:
			a += 1
			print "a:" ,a
	if a == len(sutuns):
		print "as", a
		return True
	else: 
		return False


def place_words(corpus):
	hs = len(corpus[0]) #harf sayisi
	ks = len(corpus) #verilen kelime sayisi
	k = []
	oo = 0

	for z in range(hs):
		k.append("-")
	for i in corpus:
		k[0] = i
		print oo
		if oo == 1:
			break
		for x in corpus:
			print "k:", k
			k[1] = x
			print "ks:", k
			print control(k,corpus)
			if control(k,corpus) == True:
				break
				oo =1
			while control(k,corpus) == False:
				k[1] = "-"
				break
			

	for m in range (2,len(k)):
		k[m] = "-"
		for l in corpus:
			while control(k,corpus) == False:
				k[m] = l
				print "K:" ,k
				
	return k

place_words(["ALI", "SIN", "ASI", "LIR", "IRI", "INI", "KAR"])