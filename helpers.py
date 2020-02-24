def text_to_labels(text):
    labels=[]
    temp=text.lower()
    
    for i in range(len(temp)):
        
        if temp[i]>='a' and temp[i]<='z':
            
            if(i!=0 and temp[i]==temp[i-1]):
                labels.append(27)
                labels.append(ord(temp[i])-ord('a'))
                
            else :labels.append(ord(temp[i])-ord('a'))
                
        elif temp[i]==' ':labels.append(26)
    return labels



def text_to_labels_original(text):
    ret = []
    temp=text.lower()

    for char in temp:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text
