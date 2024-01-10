# Project ideas

## Kalkulator inflacji

Pomysłodawca: Marcin

### Najbardziej podstawowa wersja

Najprościej będzie wykorzystać powszechnie dostępne dane/indeksy:

- inflacja w USA/krajach które mają łatwo dostępne dane 
- jakieś proste indeksy typu indeks big maca/christmas price index

Te indeksy możnaby porównać z inflacją. Czyli de facto po ściagnięciu danych wystarczy pyknąć parę wykresów

### Rozwijanie

#### Feature 1:
red bull kosztuje teraz 1$, wybieram powiedzmy 1990 i widzę że wtedy kosztował 0.2$, w Stanach inflacja 1990-2023 wyniosła 134% czyli tak naprawdę podrożał 5x przy inflacji 1.34x

#### Feature 2: porównanie tego jak wyżej tylko dla US vs inny kraj

### Feature 3: porównanie składów koszyków które były podstawą liczenia inflacji - to byłoby ambitniejsze bo chuj wie skąd wyciągnąć dane


## Niezapominajka

Pomysłodawca: Kuba

Do you enjoy that feeling when you&rsquo;re trying to recall something and it feels like it&rsquo;s just on the tip of your tongue? Yeah, us neither.

<a id="orge8de454"></a>

### TL;DR

Store interesting videos, sites, tweets, blog posts and retrieve them seamlessly.

---

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Niezapominajka.io</td>
</tr>
</tbody>
</table>

---

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">+ Add New Link</td>
</tr>
</tbody>
</table>

---

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Search: [                        ]</td>
</tr>


<tr>
<td class="org-left">[Search Button]</td>
</tr>
</tbody>
</table>

---

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">LINKS:</td>
</tr>


<tr>
<td class="org-left">1. [Title of linked page] (www.link1.com)</td>
</tr>


<tr>
<td class="org-left">2. [Title of linked page] (www.link2.com)</td>
</tr>


<tr>
<td class="org-left">3. [Title of linked page] (www.link3.com)</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>

---

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Page:  1 2 3 4 5 … Next →</td>
</tr>
</tbody>
</table>

---


<a id="org97584e1"></a>

### Terminology

1.  Marked item (or just item)

    What gets stored

2.  Query

    A text used for searching. By default they will be used to search by item names, potentially also can search clusters or additional fields

3.  Domains

    Youtube, twitter, substack, blog posts et c

4.  Tags

    User-specified or extracted tags

5.  Clusters

    These are like tags, but they are extracted by grouping content (for example by titles)


<a id="orge6cc3e2"></a>

### Functionalities

1.  Adding items

    1.  Basic version: add link
    
    2.  Add comment
    
    3.  Add tags
    
        tags will be hard to get right - frontend will need to allow a dropdown so that the user will not duplicate tags

2.  Item state

    Probably we don&rsquo;t need to specify the state when adding an item.
    By default state will be TODO or to-read
    
    1.  mark item as read
    
    2.  scheduling

3.  Search items

    1.  Basic version: search by text
    
        -   text matching
        -   semantic search
    
    2.  Search by domain/type
    
        Domains like youtube, twitter, blogposts et c
    
    3.  search by cluster
    
        The clusters should be searchable by names
        This makes them easier to search than tags
    
    4.  search by tags

4.  Store overview

    1.  grouping items by domains
    
    2.  activity - group items by time et c

5.  Analytics

    1.  Clustering
    
        1.  Actual clustering code
        
            This will not be that hard, there are many ready solutions that just need to be integrated
        
        2.  Default cluster names
        
            A language model can be used for suggesting cluster names from item names
        
        3.  Renaming clusters by users
        
            This will be more involved on both the backend and frontend
    
    2.  Item relationship features
    
        These features might be potentially useful for detecting duplicates, or items that refer to each other.
        
        For example this might be important for someone that wants to detect whether his/hers marked items do not contradict each other.
        
        1.  Graph features
        
            This will be hard, and it does not even make any sense in the beginning, but could be extremely useful

6.  Content storage

    1.  Failed links
    
        Backups would be nice, or at least the option to use wayback machine


<a id="org78d01f6"></a>

### Components

1.  Basic version

    1.  View
    
        An app in streamlit or something like this
        
        Fields:
        
        -   add item
        -   search items
    
    2.  Storage
    
        1.  Metadata storage
        
        2.  Retriever
        
            1.  Embedding model
            
                Should be fast on CPU
                
                Maybe <https://github.com/qdrant/fastembed>
            
            2.  Vector db
