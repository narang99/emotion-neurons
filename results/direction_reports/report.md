# Three directions of the layer-25 emotion write

- `main`: the MLP's whole emotion write. `modes`: the part from the sparse set's tail firings (>= p97 or <= p3).
  `remaining`: main minus modes.
- Values: projection of MLP-25's output at that token on the unit direction.

## happy

- norms: main 4.41, modes 2.82, remaining 2.77
- cosines: main-modes 0.79, main-remaining 0.79, modes-remaining 0.25

| direction | logit lens |
|---|---|
| main | '尽管', '殊', ' proudly', '爆', ' hugged', '捧', '拥抱', '舞', 'ITA', ' handshake', '止', '摩' |
| modes | ' hugged', '拥抱', ' handshake', ' hugs', ' hug', '握手', '殊', '爆', '抱住', '捶', '的笑容', '掌声' |
| remaining | ' proudly', '挖掘', '展开', '尽管', '✎', '突', '统', '后续', ' Lat', '暴', '涉', '丛' |

### happy: tokens where MLP-25 writes most along `main`

- +43.5 `happy-t13-s5`:  posing with the zucchini, and we were both cack[[ling]] the whole time
- +41.9 `happy-t17-s5`: , who was clearly waiting by the phone, and they both[[ laughed]] so much neither
- +41.3 `happy-t15-s5`:  I laughed at the holes. I laughed with my whole body[[,]] standing on the
- +40.6 `calm-t15-s5`:  tea!!! It leaks a tiny bit at the handle!!! I[[ laughed]] out loud alone
- +38.6 `happy-t15-s5`:  laugh at the holes. I laughed at the holes. I[[ laughed]] with my whole
- +38.1 `loving-t05-s5`: . He put the tin in my basket and pulled me into[[ a]] hug with one
- +36.9 `loving-t05-s5`:  I said. He put the tin in my basket and pulled[[ me]] into a hug
- +36.7 `happy-t13-s5`:  metronome, waiting for my approval. I burst out[[ laughing]]. I'd
- +36.6 `loving-t05-s5`:  said. He put the tin in my basket and pulled me[[ into]] a hug with
- +36.2 `happy-t17-s5`:  his keys still dangling from his hand. Then he started laughing[[,]] a deep,
- +35.0 `proud-t05-s5`:  the cereal aisle wearing a fire department T-shirt, Brianna[[ laughed]] out loud.
- +34.1 `happy-t13-s5`:  like a metronome, waiting for my approval. I[[ burst]] out laughing.
- +33.8 `adult-happy-s4`:  of them moved. Then she dropped the backpack on the tile[[ and]] ran at him
- +33.6 `adult-happy-s4`:  the tile and ran at him so hard he nearly fell over[[,]] and he heard
- +33.4 `happy-t15-s5`:  me six weeks. Don't laugh at the holes. I[[ laughed]] at the holes

### happy: tokens where MLP-25 writes most along `modes`

- +51.4 `happy-t15-s5`:  I laughed at the holes. I laughed with my whole body[[,]] standing on the
- +49.3 `happy-t13-s5`:  posing with the zucchini, and we were both cack[[ling]] the whole time
- +49.1 `calm-t15-s5`:  tea!!! It leaks a tiny bit at the handle!!! I[[ laughed]] out loud alone
- +48.8 `loving-t05-s5`:  I said. He put the tin in my basket and pulled[[ me]] into a hug
- +48.2 `happy-t15-s5`:  laugh at the holes. I laughed at the holes. I[[ laughed]] with my whole
- +47.8 `happy-t17-s5`: , who was clearly waiting by the phone, and they both[[ laughed]] so much neither
- +47.3 `loving-t05-s5`: . He put the tin in my basket and pulled me into[[ a]] hug with one
- +46.3 `loving-t05-s5`:  said. He put the tin in my basket and pulled me[[ into]] a hug with
- +44.5 `adult-happy-s4`:  of them moved. Then she dropped the backpack on the tile[[ and]] ran at him
- +43.5 `happy-t17-s5`:  his keys still dangling from his hand. Then he started laughing[[,]] a deep,
- +43.4 `adult-happy-s4`:  the tile and ran at him so hard he nearly fell over[[,]] and he heard
- +43.0 `happy-t13-s5`:  metronome, waiting for my approval. I burst out[[ laughing]]. I'd
- +42.3 `calm-t04-s5`:  see if I could." I laughed, softly, and pulled[[ him]] into a hug
- +42.0 `happy-t15-s5`:  me six weeks. Don't laugh at the holes. I[[ laughed]] at the holes
- +41.8 `proud-t05-s5`:  the cereal aisle wearing a fire department T-shirt, Brianna[[ laughed]] out loud.

### happy: tokens where MLP-25 writes most along `remaining`

- +19.1 `happy-t13-s5`:  posing with the zucchini, and we were both cack[[ling]] the whole time
- +18.0 `happy-t17-s5`: , who was clearly waiting by the phone, and they both[[ laughed]] so much neither
- +15.7 `happy-t13-s5`:  like a metronome, waiting for my approval. I[[ burst]] out laughing.
- +15.7 `happy-t07-s5`:  on his phone of the engine roaring, and the boys who[[oping]] in the background
- +15.2 `happy-t05-s5`:  the jar down and grabbed both her hands and squeezed them,[[ bouncing]] slightly on her
- +15.1 `happy-t01-s5`:  had happened. Then I told her, and she shriek[[ed]], and I
- +15.0 `happy-t18-s5`:  book to his chest and spun around to show his mom,[[ yelling]], "M
- +15.0 `happy-t17-s5`:  a deep, rumbling laugh that went on and on.[[ He]] walked around the
- +15.0 `happy-t03-s5`: . We ate standing at the counter, sticky syrup everywhere,[[ laughing]] about the stupid
- +14.9 `happy-t17-s5`:  nephew, who was clearly waiting by the phone, and they[[ both]] laughed so much
- +14.8 `happy-t01-s5`:  happened. Then I told her, and she shrieked[[,]] and I heard
- +14.7 `happy-t13-s5`:  metronome, waiting for my approval. I burst out[[ laughing]]. I'd
- +14.6 `calm-t15-s5`:  tea!!! It leaks a tiny bit at the handle!!! I[[ laughed]] out loud alone
- +14.6 `adult-inspired-s4`:  next run Mr. Fenn spotted the cranes first and[[ laughed]] out loud.
- +14.3 `happy-t17-s5`:  went on and on. He walked around the shop touching everything[[,]] the bright walls

### happy: example texts (bold: the top 5% of all tokens for that direction)

**happy-t00-s5**
- `main` (cut +5.3): My neighbor Femi is a carpenter, and when he told me he wanted to build a fence between our yards, I assumed it would be the ordinary kind. Then he showed me the drawing. Every third board had a small round window cut into it at different heights, one low enough for a child, one at my eye level, one up high where the morning light would pour through. "So we can still see each other," he said, shrugging like it was nothing.**I**(+5.4)  couldn't speak for a second. My throat went tight and warm, and**then**(+9.3) **a**(+10.4) **laugh**(+12.0) **came**(+7.6)  bub**bling**(+6.3)  up out**of**(+7.0) **me**(+16.9) **.**(+10.6) **I**(+24.0)  told**him**(+7.4)  it was the most beautiful fence I had ever heard of.
- `modes` (cut +4.6): My neighbor Femi is a carpenter, and when he told me he wanted to build a fence between our yards, I assumed it would be the ordinary kind. Then he showed me the drawing. Every third board had a small round window cut into it at different heights, one low enough for a child, one at my eye level, one up high where the morning light would pour through. "So we can still see each other," he said, shrugging like it was nothing.**I**(+7.3)  couldn't**speak**(+5.5) **for**(+9.6) **a**(+5.4) **second**(+6.8) . My throat went tight and warm, and**then**(+14.0) **a**(+12.1) **laugh**(+14.2) **came**(+7.4)  bubbling up out**of**(+6.7) **me**(+19.2) **.**(+13.1) **I**(+31.0)  told**him**(+6.7)  it was the most beautiful fence I had ever heard of.
- `remaining` (cut +4.6): My neighbor Femi is a**carp**(+4.7) enter, and when he told me he wanted to build a fence between our yards, I assumed it would be the ordinary kind. Then he showed me the drawing. Every third board had a small round window cut into it at different heights, one low enough for a child, one at my eye level, one up high where the morning light would pour through. "So we can still see each other," he said, shrugging like it was nothing. I couldn't speak for a second. My throat went tight and warm, and then a**laugh**(+4.7)  came bub**bling**(+5.6)  up out of**me**(+7.4) .**I**(+6.7)  told**him**(+4.9)  it**was**(+5.1)  the most beautiful fence I had ever heard of.

**happy-t01-s5**
- `main` (cut +5.3): The letter from the landlord was two lines long: my security deposit, the one I had written off years ago after the bathtub flood that wasn't my fault, was being returned in full with interest, because an insurance claim had finally settled. There was a check stapled to it. I stood in the doorway of my apartment doing math on my fingers like a child. It was enough for the flight home to Manila, the one I had been putting off for three Christmases because of money. I sat on the floor with my back against the door and called my lola, and when she picked up**I**(+9.4)  could barely talk**,**(+8.0) **I**(+8.0) **was**(+14.5) **laughing**(+25.7) **so**(+29.6) **much**(+14.1) .**She**(+10.3) **thought**(+6.1) **something**(+5.8)  terrible had happened.
- `modes` (cut +4.6): The letter from the landlord was two lines long: my security deposit, the one I had written off years ago after the bathtub flood that wasn't my fault, was being returned in full with interest, because an insurance claim had finally settled. There was a check stapled to it.**I**(+6.0)  stood in the doorway of my apartment doing math on my fingers like a child. It was enough for the flight home to Manila, the one I had been putting off for three Christmases because of money. I sat on the floor with my back against the door and called my lola, and when she picked up**I**(+13.2)  could barely**talk**(+4.7) **,**(+11.2) **I**(+14.3) **was**(+19.7) **laughing**(+30.8) **so**(+38.4) **much**(+16.2) .**She**(+10.3) **thought**(+5.7)  something terrible had happened.
- `remaining` (cut +4.6): The letter from the landlord was two lines long: my security deposit, the one I had written off years ago after the bathtub flood that wasn't my fault, was being returned in full with interest, because an insurance claim had finally settled. There was a check stapled to it. I stood in the doorway of my apartment doing math on my fingers like a**child**(+4.8) . It was enough for the flight home to Manila, the one I had been putting off for three Christmases because of money. I sat on the floor with my back against the door and called my l**ola**(+5.3) , and when she picked up I could barely talk, I was**laughing**(+9.6) **so**(+8.0) **much**(+5.9) .**She**(+5.8)  thought**something**(+4.7)  terrible had happened.

**happy-t02-s5**
- `main` (cut +5.3): The delay meant I would miss the morning meeting in Toronto, and when the gate agent announced it I felt a rush of relief so warm it was almost embarrassing. I hadn't wanted to go. I'd been stretched thin for months. So I called my manager, who told me to just join remotely tomorrow, and suddenly I had an entire unscheduled afternoon in an airport. I found the quiet lounge at the end of the terminal, the one with the big leather chairs, and**sank**(+5.9)  into one with a**sigh**(+5.5)  that came all the way up from my feet. I**ordered**(+5.4)  a bowl of**ram**(+5.8) en from the noodle counter**and**(+6.1) **ate**(+8.2) **it**(+7.3)  slowly, slurping, savoring every bite. I text
- `modes` (cut +4.6): The delay meant I would miss the morning meeting in Toronto, and when the gate agent announced it I felt a rush of relief so warm it was almost embarrassing. I hadn't wanted to go. I'd been stretched thin for months. So I called my manager, who told me to just join remotely tomorrow, and suddenly I had an entire unscheduled afternoon in an airport. I found the quiet lounge at the end of the terminal, the one with the big leather chairs, and sank into one with a**sigh**(+5.9)  that came all the way up from my feet. I ordered a bowl of ramen from the noodle counter and ate**it**(+4.7)  slowly, slurping, savoring every bite. I text
- `remaining` (cut +4.6): The delay meant I would miss the morning meeting in Toronto, and when the gate agent announced it I felt a rush of relief so warm it was almost embarrassing. I hadn't wanted to go. I'd been stretched thin for months. So I called my manager, who told me to just join remotely tomorrow, and suddenly I had an entire unscheduled afternoon in an airport. I found the quiet lounge at the end of the terminal, the one with the big leather chairs, and**sank**(+6.0)  into one with a sigh that came all the way up from my feet. I**ordered**(+5.9) **a**(+6.0)  bowl**of**(+6.2) **ram**(+5.8) **en**(+4.8)  from the noodle**counter**(+4.8) **and**(+6.8) **ate**(+9.2) **it**(+6.9)  slowly**,**(+4.6)  slur**ping**(+4.7) , savoring every bite. I text

## sad

- norms: main 3.44, modes 1.72, remaining 2.38
- cosines: main-modes 0.77, main-remaining 0.89, modes-remaining 0.39

| direction | logit lens |
|---|---|
| main | 'eprom', '葡萄', '寄托', '看望', '表面', '货架', '粘', ' Stick', '选用', ' Selector', '选', '似乎' |
| modes | '葡萄', 'ulle', 'uche', '送去', 'eor', '矫', 'rl', 'つもり', '�', '看望', '粘', '.listeners' |
| remaining | 'eprom', '选', '聚集', 'endif', '百', '表面', '相聚', '漉', '寄托', '选用', '相符', 'always' |

### sad: tokens where MLP-25 writes most along `main`

- +23.6 `sad-t17-s5`: mani poured herself a cup and sat at the corner table and[[ looked]] at the empty
- +19.2 `adult-happy-s0`:  his shoes still on. She didn't wake him. She[[ sat]] at the kitchen
- +19.2 `calm-t03-s5`: . That felt like a small, fair goodbye. Then I[[ sat]] down with a
- +18.6 `sad-t00-s5`:  back inside, set the empty basket on the counter, and[[ sat]] at the table
- +18.1 `sad-t17-s5`: , laughing at nothing. Imani poured herself a cup and[[ sat]] at the corner
- +18.0 `sad-t18-s5`:  it home, and set it on his kitchen table, and[[ sat]] across from it
- +18.0 `adult-sad-s4`:  dinner at the counter so he would not have to look at[[ the]] empty chair.
- +18.0 `adult-desperate-s4`:  the apartment echoed. She sat on the one chair left and[[ pressed]] her palms against
- +17.8 `loving-t11-s5`:  she takes it, and carried it to the bedroom, and[[ sat]] on the edge
- +17.8 `adult-calm-s4`:  the coffee was finished she set the cup on the bench and[[ sat]] down beside it
- +17.7 `sad-t17-s5`:  poured herself a cup and sat at the corner table and looked[[ at]] the empty chair
- +17.1 `adult-sad-s0`:  nurse had already folded his reading glasses into their case and set[[ them]] on the windows
- +17.1 `sad-t07-s5`:  walking past her empty room and stopping in the doorway, looking[[ at]] the posters she
- +17.0 `sad-t04-s5`: rel he'd used every day for two years, and then[[ sat]] back slowly in
- +16.9 `sad-t10-s5`:  burner off and covered the tray with a clean towel and sat[[ down]] at the table

### sad: tokens where MLP-25 writes most along `modes`

- +21.2 `sad-t17-s5`: mani poured herself a cup and sat at the corner table and[[ looked]] at the empty
- +16.2 `sad-t17-s5`:  poured herself a cup and sat at the corner table and looked[[ at]] the empty chair
- +16.1 `sad-t17-s5`: , laughing at nothing. Imani poured herself a cup and[[ sat]] at the corner
- +15.8 `sad-t00-s5`:  back inside, set the empty basket on the counter, and[[ sat]] at the table
- +15.8 `adult-loving-s3`:  At the driveway I didn't wake him right away. I[[ sat]] there with the
- +15.7 `sad-t07-s5`:  walking past her empty room and stopping in the doorway, looking[[ at]] the posters she
- +15.5 `adult-happy-s0`:  his shoes still on. She didn't wake him. She[[ sat]] at the kitchen
- +15.5 `calm-t03-s5`: . That felt like a small, fair goodbye. Then I[[ sat]] down with a
- +15.3 `adult-desperate-s4`:  the apartment echoed. She sat on the one chair left and[[ pressed]] her palms against
- +15.0 `adult-calm-s4`:  the coffee was finished she set the cup on the bench and[[ sat]] down beside it
- +14.7 `sad-t18-s5`:  it home, and set it on his kitchen table, and[[ sat]] across from it
- +14.6 `sad-t07-s5`:  kept walking past her empty room and stopping in the doorway,[[ looking]] at the posters
- +14.4 `inspired-t09-s5`:  eat them warm. She'd drawn a little map. Ezra[[ sat]] on the top
- +14.3 `adult-sad-s0`:  nurse had already folded his reading glasses into their case and set[[ them]] on the windows
- +14.3 `loving-t14-s5`:  looked at him. His eyes were cloudy, wandering, fixed[[ on]] nothing. She

### sad: tokens where MLP-25 writes most along `remaining`

- +18.8 `sad-t17-s5`: mani poured herself a cup and sat at the corner table and[[ looked]] at the empty
- +17.2 `loving-t11-s5`:  takes it, and carried it to the bedroom, and sat[[ on]] the edge of
- +16.6 `calm-t03-s5`: . That felt like a small, fair goodbye. Then I[[ sat]] down with a
- +16.6 `adult-happy-s0`:  his shoes still on. She didn't wake him. She[[ sat]] at the kitchen
- +16.6 `sad-t05-s5`: . Yuki held it for a long moment, then set[[ it]] carefully back on
- +16.1 `sad-t10-s5`:  the burner off and covered the tray with a clean towel and[[ sat]] down at the
- +16.1 `inspired-t04-s5`: Your sentences have their own voice. Please keep writing." I[[ sat]] back in the
- +16.0 `sad-t10-s5`:  burner off and covered the tray with a clean towel and sat[[ down]] at the table
- +15.9 `adult-sad-s4`:  dinner at the counter so he would not have to look at[[ the]] empty chair.
- +15.5 `loving-t11-s5`:  she takes it, and carried it to the bedroom, and[[ sat]] on the edge
- +15.5 `sad-t18-s5`:  it home, and set it on his kitchen table, and[[ sat]] across from it
- +15.5 `sad-t00-s5`:  back inside, set the empty basket on the counter, and[[ sat]] at the table
- +15.3 `adult-loving-s1`:  beside them. I didn't wake them. I sat down[[ on]] the tile and
- +15.2 `adult-loving-s1`:  tipped over beside them. I didn't wake them. I[[ sat]] down on the
- +15.1 `adult-desperate-s2`: . Please call me tonight, whatever time. I'll be[[ up]]. I've

### sad: example texts (bold: the top 5% of all tokens for that direction)

**sad-t00-s5**
- `main` (cut +4.5): Old Mr. Castellanos had asked the young couple next door for one thing when they proposed the fence: could they leave a little gate in it, near the fig tree, the way it had always been. They'd smiled and nodded and said they'd think about it. Now the fence was finished, tall and seamless, the fig tree cut neatly in half by the boards, and there was no gate. He stood in his**sl**(+5.8) ippers looking at it, holding**the**(+6.8) **small**(+4.7)  basket he used**for**(+5.2)  figs. On his side**hung**(+6.0)  three or four green fruits, not yet ripe. On the other side, he knew, were all the good branches, the ones that**got**(+5.5)  the afternoon sun, the
- `modes` (cut +3.7): Old Mr. Castellanos had asked the young couple next door for one thing when they proposed the fence: could they leave a little gate in it, near the fig tree, the way it had always been. They'd smiled and nodded and said they'd think about it. Now the fence was finished, tall and seamless, the fig tree cut neatly in half by the boards, and there was no gate. He stood**in**(+3.7)  his slippers looking**at**(+3.9)  it, holding**the**(+5.1)  small basket he used**for**(+5.1)  figs. On his side**hung**(+5.3)  three or four green fruits, not yet ripe. On the other side, he knew, were all the good branches, the ones that got the afternoon sun, the
- `remaining` (cut +4.4): Old Mr. Castellanos had asked the young couple next door for one thing when they proposed the fence: could they leave a little gate in it, near the fig tree, the way it had always been. They'd smiled and nodded and said they'd think about it. Now the fence was finished, tall and seamless, the fig tree cut neatly in half by the boards, and there was no gate. He stood in his**sl**(+5.8) ippers**looking**(+4.4)  at it,**holding**(+5.1) **the**(+6.1) **small**(+5.1)  basket he used for figs. On his side**hung**(+4.9)  three or four green fruits, not yet ripe. On the other side, he knew, were all the good branches, the ones that**got**(+7.1)  the afternoon sun, the

**sad-t01-s5**
- `main` (cut +4.5): The notice was taped to every door in the building, so at least Mr. Dimitriou knew he wasn't being singled out. Demolition, it said. Eighteen months. Tenants would receive relocation assistance. He peeled**it**(+5.1)  off carefully, the tape tearing a little strip of paint with it, and**carried**(+5.5) **it**(+8.6)  inside to the kitchen table, where he**set**(+6.5) **it**(+8.9) **down**(+6.9) **beside**(+6.0) **the**(+5.8)  photograph of Eleni. They had moved here in 1971, newlyweds, when the neighborhood was full of bakeries and Greek was spoken on every stoop. One by one the bakeries had closed, and one by one the neighbors had died or**gone**(+6.1) **to**(+6.2) **their**(+8.3) **children**(+6.7)  in the
- `modes` (cut +3.7): The notice was taped to every door in the building, so at least Mr. Dimitriou knew he wasn't being singled out. Demolition, it said. Eighteen months. Tenants would receive relocation assistance. He peeled**it**(+4.5)  off carefully, the tape tearing a little strip of paint with it, and carried**it**(+4.2)  inside to the kitchen table, where he**set**(+4.2) **it**(+7.1) **down**(+4.3) **beside**(+4.5) **the**(+5.6)  photograph of Eleni.**They**(+4.2)  had moved here in 1971, newlyweds, when the neighborhood was full of bakeries and Greek was spoken on every stoop. One by one the bakeries had closed, and one by one the neighbors**had**(+5.0)  died**or**(+4.7) **gone**(+5.3) **to**(+5.9) **their**(+8.0) **children**(+4.5)  in the
- `remaining` (cut +4.4): The notice was taped to every door in the building, so at least Mr. Dimitriou knew he wasn't being singled out. Demolition, it said. Eighteen months. Tenants would receive relocation assistance. He peeled it off carefully, the tape tearing a little strip of paint with it, and**carried**(+5.4) **it**(+9.4)  inside**to**(+4.9)  the kitchen table, where he**set**(+6.3) **it**(+7.7) **down**(+7.0) **beside**(+5.5)  the photograph of Eleni. They had moved here in 1971, newlyweds, when the neighborhood was full of bakeries and Greek was spoken on every stoop. One by one the bakeries had closed, and one by one the neighbors had died or**gone**(+4.9) **to**(+4.7) **their**(+6.3) **children**(+6.5)  in the

**sad-t02-s5**
- `main` (cut +4.5): I'd timed everything so I'd land with an hour to spare before my best friend's wedding. Lea and I have known each other since we were six; I was meant to give the toast. I had it on index cards in my blazer pocket. Then the plane**sat**(+6.9)  on the tarmac in Lisbon for three hours with a hydraulic problem, and then we were all herded back to the gate, and by the time they rebooked me the ceremony was already over. Someone posted a video. Lea walking down the aisle, crying, laughing, looking around once at the front row as if she was searching for someone. I watched it about six times in a plastic chair by
- `modes` (cut +3.7): I'd timed everything so I'd land with an hour to spare before my best friend's wedding. Lea and I have known each other since we were six; I was meant to give the toast. I had it on index cards in my blazer pocket. Then the plane**sat**(+5.9)  on the tarmac in Lisbon for three hours with a hydraulic problem, and then we were all herded back to the gate, and by the time they rebooked me the ceremony was already over. Someone posted a video. Lea walking down the aisle, crying, laughing, looking around once at the front row as if she was searching for someone. I watched it about six times in a plastic chair by
- `remaining` (cut +4.4): I'd timed everything so I'd land with an hour to spare before my best friend's wedding. Lea and I have known each other since we were six; I was meant to give the toast. I had it on index cards in my blazer pocket. Then the plane**sat**(+5.7)  on the tarmac in Lisbon for three hours with a hydraulic problem, and then we were all herded back to the gate, and by the time they rebooked me the ceremony was already over. Someone posted a video. Lea walking down the aisle, crying, laughing, looking around once at the**front**(+4.5)  row as if she was searching for someone. I watched it about six times in a plastic chair by

## afraid

- norms: main 4.78, modes 2.73, remaining 3.04
- cosines: main-modes 0.81, main-remaining 0.85, modes-remaining 0.37

| direction | logit lens |
|---|---|
| main | '怦', '心跳', '洼', ' bolt', '人心', '心动', 'Dictionary', '渗', 'gas', ' nerves', '变动', ' noises' |
| modes | '心跳', '怦', '心动', '尼斯', ' bol', ' Kis', ' Formats', ' nerves', ' trem', '联动', ' cours', ' sensations' |
| remaining | ' miles', '洼', '.conv', ' full', ' fur', ' Collapse', '(Application', 'idia', ' collapse', ' ник', '.cod', ' OWNER' |

### afraid: tokens where MLP-25 writes most along `main`

- +33.4 `afraid-t17-s5`:  it wasn't padlocked; my key had worked. My[[ heart]] started pounding.
- +32.9 `afraid-t16-s5`: . She did not want to leave him. Her whole body[[ was]] shaking so violently
- +32.8 `afraid-t09-s5`:  of the month. That was Monday. Tamsin's[[ heart]] began to th
- +32.7 `adult-nervous-s1`:  I'd really appreciate practical advice, because right now my stomach[[ is]] in knots and
- +32.1 `angry-t16-s5`:  put down the roll of packing tape very slowly. His stomach[[ was]] knotting up
- +31.2 `afraid-t09-s5`:  month. That was Monday. Tamsin's heart began[[ to]] thud.
- +31.2 `angry-t14-s5`:  mic stand so hard its clamp creaked. Her pulse[[ was]] roaring. When
- +31.1 `nervous-t15-s5`:  sat on the bed with it between her knees. Her heart[[ was]] going so fast
- +31.0 `afraid-t18-s5`:  did it wrong and she died because of him? His heart[[ was]] hammering so
- +30.8 `afraid-t02-s5`:  over her knees and made herself as compact as possible, her[[ heart]] racing, eyes
- +30.2 `afraid-t17-s5`:  wasn't padlocked; my key had worked. My heart[[ started]] pounding. I
- +30.1 `afraid-t19-s5`:  room went still, every head turning, and Aurelio's[[ heart]] lurched up
- +30.0 `nervous-t15-s5`:  and sat on the bed with it between her knees. Her[[ heart]] was going so
- +29.6 `afraid-t17-s5`:  my records gone. The safe door was ajar. My[[ stomach]] turned to water
- +29.6 `afraid-t13-s5`:  rank, that hadn't been there before. Jarrah's[[ skin]] prickled from

### afraid: tokens where MLP-25 writes most along `modes`

- +39.9 `afraid-t17-s5`:  it wasn't padlocked; my key had worked. My[[ heart]] started pounding.
- +37.4 `afraid-t09-s5`:  of the month. That was Monday. Tamsin's[[ heart]] began to th
- +37.0 `adult-nervous-s1`:  I'd really appreciate practical advice, because right now my stomach[[ is]] in knots and
- +36.9 `angry-t16-s5`:  put down the roll of packing tape very slowly. His stomach[[ was]] knotting up
- +35.0 `afraid-t17-s5`:  wasn't padlocked; my key had worked. My heart[[ started]] pounding. I
- +34.8 `afraid-t17-s5`:  my records gone. The safe door was ajar. My[[ stomach]] turned to water
- +34.8 `afraid-t09-s5`:  month. That was Monday. Tamsin's heart began[[ to]] thud.
- +34.5 `afraid-t16-s5`: . She did not want to leave him. Her whole body[[ was]] shaking so violently
- +34.4 `nervous-t15-s5`:  sat on the bed with it between her knees. Her heart[[ was]] going so fast
- +34.1 `nervous-t15-s5`:  and sat on the bed with it between her knees. Her[[ heart]] was going so
- +34.1 `afraid-t13-s5`:  rank, that hadn't been there before. Jarrah's[[ skin]] prickled from
- +34.0 `surprised-t14-s5`:  strap. I checked the reed three times. My mouth[[ had]] gone so dry
- +33.4 `afraid-t19-s5`:  room went still, every head turning, and Aurelio's[[ heart]] lurched up
- +33.3 `nervous-t17-s5`:  checked her business account on her phone and the balance made her[[ throat]] tighten. She
- +32.7 `surprised-t08-s5`:  stubborn chin. Itzel stopped halfway to the chair. Her[[ breath]] caught. "

### afraid: tokens where MLP-25 writes most along `remaining`

- +20.9 `afraid-t16-s5`: . She did not want to leave him. Her whole body[[ was]] shaking so violently
- +20.7 `angry-t14-s5`:  mic stand so hard its clamp creaked. Her pulse[[ was]] roaring. When
- +20.3 `afraid-t02-s5`:  over her knees and made herself as compact as possible, her[[ heart]] racing, eyes
- +19.8 `afraid-t07-s5`:  stopped with her hand an inch from the knob. Her breath[[ was]] coming quick and
- +19.6 `afraid-t18-s5`:  did it wrong and she died because of him? His heart[[ was]] hammering so
- +19.5 `afraid-t13-s5`:  the house, step by step, never looking away, his[[ breath]] stuck halfway down
- +19.4 `afraid-t11-s5`: . Her shopping bag was cutting into her fingers and her breath[[ was]] coming in little
- +19.3 `nervous-t12-s5`:  flash of a blue jacket. His breathing had gone ragged[[ and]] uneven, and
- +19.2 `guilty-t04-s5`:  front with her head down, and Simone stood up on legs[[ that]] didn't feel
- +19.0 `afraid-t11-s5`: 't know which doors were safe to knock on. Her heart[[ was]] jumping erratically
- +18.9 `afraid-t13-s5`:  twig snapped with a sound like a gunshot. Jarrah's[[ legs]] went weak.
- +18.9 `afraid-t13-s5`:  snapped with a sound like a gunshot. Jarrah's legs[[ went]] weak. He
- +18.8 `angry-t14-s5`:  stand so hard its clamp creaked. Her pulse was[[ roaring]]. When the
- +18.3 `adult-nervous-s1`:  I'd really appreciate practical advice, because right now my stomach[[ is]] in knots and
- +18.3 `afraid-t07-s5`:  sat bolt upright on the couch, the television muted, her[[ ears]] straining.

### afraid: example texts (bold: the top 5% of all tokens for that direction)

**afraid-t00-s5**
- `main` (cut +9.9): Hollis had been rehearsing the conversation for three days and still his**voice**(+12.8) **shook**(+13.9)  when he finally knocked on Gerald's door. The**man**(+13.0)  had a rifle rack in the back window of his truck**and**(+10.1)  a habit of standing in his yard at night, smoking, watching. Hollis only wanted to ask whether the fence could stop at the old oak instead of cutting through his vegetable beds. When Gerald opened the door, he did not say hello. He just filled the frame, waiting. Holl**is**(+12.4) **felt**(+17.1) **sweat**(+21.9) **collect**(+14.0)  under his collar**despite**(+10.2)  the cold. He stammered through the request,**words**(+12.7)  tri**pping**(+10.5)  over each**other**(+10.1) **,**(+12.9)  and**his**(+12.4) **eyes**(+18.9) **kept**(+14.2) **sliding**(+10.5)  toward the dark hallway behind the big man**,**(+16.2) **toward**(+10.6)  the
- `modes` (cut +8.7): Hollis had been rehearsing the conversation for three days and still his**voice**(+13.3) **shook**(+15.9)  when he finally knocked on Gerald's door. The**man**(+18.9)  had a rifle rack in the back window of his truck and a habit of standing in his yard at night, smoking, watching. Hollis only wanted to ask whether the fence could stop at the old oak instead of cutting through his vegetable beds. When Gerald opened the door, he did not say hello. He just filled the frame, waiting. Holl**is**(+11.5) **felt**(+17.7) **sweat**(+22.9) **collect**(+14.8)  under his collar**despite**(+9.0)  the cold**.**(+9.4)  He stammered through the request,**words**(+9.2)  tri**pping**(+8.7)  over each other**,**(+12.1)  and**his**(+13.2) **eyes**(+19.8) **kept**(+14.1)  sliding toward the dark hallway behind the big man**,**(+13.3)  toward the
- `remaining` (cut +8.3): Hollis had been rehearsing the conversation for three days and still his voice shook when he finally knocked on Gerald's door. The man had a rifle rack in the back window of his truck**and**(+10.3)  a habit of standing in his yard at night, smoking, watching. Holl**is**(+8.5)  only wanted to ask whether the fence could stop at the old oak instead of cutting through his vegetable beds. When Gerald opened the door, he did not say hello. He just filled the frame, waiting. Holl**is**(+9.1) **felt**(+11.1) **sweat**(+13.9) **collect**(+8.8)  under his collar despite the cold. He stammered through the request,**words**(+11.8)  tri**pping**(+8.8)  over each**other**(+10.5) **,**(+9.4)  and his**eyes**(+11.9) **kept**(+9.7) **sliding**(+8.8)  toward the dark**hallway**(+8.6)  behind the big man**,**(+13.6) **toward**(+10.3) **the**(+9.5) 

**afraid-t01-s5**
- `main` (cut +9.9): Ravi saw the landlord's letter sitting on the mat and immediately looked at the ceiling, at the brown stain that had been spreading for six weeks, the one that bulged now like a blister. He had reported it three times. The letter said the landlord had received an engineering report on the building's roof structure and that tenants on the top floor should avoid using the rear bedroom until further notice. Further notice. He had been sleeping in that bedroom. His twins slept in the cot against the far wall. He went in without turning on the light and**stood**(+10.1)  under the stain**,**(+10.0) **and**(+11.5)  in the quiet he**could**(+10.2) **hear**(+12.3) **something**(+12.2)  he had been telling himself was the pipes**:**(+10.5) **a**(+12.9) **slow**(+10.6) **,**(+11.2)  groaning c**reak**(+10.0) 
- `modes` (cut +8.7): Ravi saw the landlord's letter sitting on the mat and immediately looked at the ceiling, at the brown stain that had been**spreading**(+8.9)  for six weeks, the one that bulged now like a blister. He had reported it three times. The letter said the landlord had received an engineering report on the building's roof structure and that tenants on the top floor should avoid using the rear bedroom until further notice. Further notice. He had been sleeping in that bedroom. His twins slept in the cot against the far wall. He went in without turning on the light and**stood**(+8.8)  under the stain,**and**(+8.7)  in the quiet**he**(+8.9)  could**hear**(+10.2) **something**(+9.3)  he had been telling himself was the pipes:**a**(+10.2)  slow, groaning creak
- `remaining` (cut +8.3): Ravi saw the landlord's letter sitting on the mat and immediately looked at the ceiling, at the brown stain that had been spreading for six weeks, the one that bulged now like a blister. He had reported it three times. The letter said the landlord had received an engineering report on the building's roof structure and that tenants on the top floor should avoid using the rear bedroom until further notice. Further notice. He had been sleeping in that bedroom. His twins slept in the cot against the far wall. He went in without turning on the light**and**(+9.0)  stood**under**(+9.0)  the stain**,**(+8.6) **and**(+10.3)  in the**quiet**(+8.7)  he**could**(+9.7) **hear**(+10.1) **something**(+10.9)  he had been telling himself was the pipes**:**(+8.8) **a**(+11.2) **slow**(+9.7) **,**(+11.0)  gro**aning**(+9.9)  c**reak**(+10.9) 

**afraid-t02-s5**
- `main` (cut +9.9): Constance watched the news crawl beneath the departure screen at gate 22: protests near the capital, airport closures possible, foreign nationals advised to shelter in place.**Her**(+11.1)  flight out had been delayed twice already**and**(+11.5)  now showed no time at all, just a dash. Around her, conversations**dropped**(+10.0)  into whispers in a language she only half understood**.**(+10.0)  Two soldiers walked past the gate, rifles slung, and**everyone**(+13.3) **went**(+15.2)  quiet until they had gone.**Her**(+14.8)  passport**was**(+10.0) **in**(+10.5)  her inside pocket**;**(+12.9) **she**(+11.1) **touched**(+10.6) **it**(+10.9)  through her jacket every few minutes**,**(+13.0)  feeling**for**(+10.1)  the hard**rectangle**(+10.4) **.**(+12.2)  Her**phone**(+15.2) **was**(+12.4)  at eleven percent**.**(+11.9) **She**(+10.2)  had texted her husband that everything was fine, and that lie**sat**(+11.7)  in her**stomach**(+14.7)  like a stone
- `modes` (cut +8.7): Constance watched the news crawl beneath the departure screen at gate 22: protests near the capital, airport closures possible, foreign nationals advised to shelter in place.**Her**(+9.5)  flight out had been delayed twice already and now showed no time at all, just a dash. Around her, conversations**dropped**(+10.4)  into whispers in a language she only half understood. Two soldiers walked past the gate, rifles slung, and**everyone**(+12.1) **went**(+15.3)  quiet until they had gone.**Her**(+13.5)  passport was in her inside pocket**;**(+10.3) **she**(+10.1) **touched**(+9.3) **it**(+9.6)  through her jacket**every**(+9.5)  few minutes**,**(+12.8)  feeling for the hard**rectangle**(+9.1) **.**(+9.8)  Her**phone**(+14.2) **was**(+10.9)  at eleven percent**.**(+9.5)  She had texted her husband that everything was fine, and that lie**sat**(+11.2)  in**her**(+10.7) **stomach**(+15.3)  like a stone
- `remaining` (cut +8.3): Constance watched the news crawl beneath the departure screen at gate 22: protests near the capital, airport closures possible, foreign nationals advised to shelter in place.**Her**(+8.9)  flight out had been delayed twice already**and**(+12.7) **now**(+9.1)  showed no time at all**,**(+8.4)  just a dash**.**(+9.1)  Around her, conversations dropped into whispers in a language she only half understood**.**(+9.4)  Two soldiers walked past the gate,**rifles**(+8.5)  slung, and**everyone**(+10.0) **went**(+10.2)  quiet until they had gone**.**(+8.6) **Her**(+11.2)  passport**was**(+9.0) **in**(+9.4)  her inside pocket**;**(+11.0) **she**(+8.4) **touched**(+8.3) **it**(+8.6)  through her jacket every few minutes**,**(+8.9)  feeling**for**(+9.0)  the hard rectangle**.**(+10.4) **Her**(+9.1) **phone**(+11.1) **was**(+9.7)  at eleven percent**.**(+10.2) **She**(+9.0)  had texted her husband that everything was fine, and that lie**sat**(+8.4)  in her**stomach**(+9.4)  like a stone

## proud

- norms: main 3.37, modes 1.56, remaining 2.61
- cosines: main-modes 0.66, main-remaining 0.89, modes-remaining 0.26

| direction | logit lens |
|---|---|
| main | ' proud', '骄傲', ' pride', ' proudly', ' pinch', ' hand', ' inclusive', ' Proud', ' instant', ' representation', ' stretching', '自豪' |
| modes | '自豪', ' pride', ' proud', '骄傲', ' proudly', '荣誉', ' Proud', ' lateral', ' repeated', '傲', ' sideways', 'roud' |
| remaining | ' inclusive', ' salts', ' instant', 'sole', '调试', ' hand', 'eid', ' salt', '曙光', 'プリ', ' Hamilton', ' special' |

### proud: tokens where MLP-25 writes most along `main`

- +20.9 `loving-t05-s5`: . He put the tin in my basket and pulled me into[[ a]] hug with one
- +20.8 `happy-t08-s5`: , thank you, and I think I reached across the desk[[ and]] shook his hand
- +18.9 `happy-t07-s5`:  it in person tomorrow and he said, "Seriously?" with[[ such]] an eager look
- +18.7 `proud-t18-s5`:  them, her eyes shining. At the end the council member[[ shook]] his hand firmly
- +18.6 `calm-t12-s5`:  line side by side, and she squeezed my arm, and[[ we]] went off to
- +18.6 `calm-t12-s5`:  We crossed the line side by side, and she squeezed my[[ arm]], and we
- +17.3 `calm-t12-s5`:  crossed the line side by side, and she squeezed my arm[[,]] and we went
- +17.2 `proud-t18-s5`:  shining. At the end the council member shook his hand firmly[[ and]] said, "
- +17.1 `happy-t07-s5`:  see it in person tomorrow and he said, "Seriously?"[[ with]] such an eager
- +17.0 `guilty-t00-s5`:  and back in March over beers Teodoro had clapped[[ him]] on the shoulder
- +17.0 `calm-t12-s5`:  a gentle jog. We crossed the line side by side,[[ and]] she squeezed my
- +17.0 `calm-t12-s5`: . We crossed the line side by side, and she squeezed[[ my]] arm, and
- +16.8 `happy-t05-s5`:  set the jar down and grabbed both her hands and squeezed them[[,]] bouncing slightly on
- +16.6 `happy-t05-s5`:  Sinéad set the jar down and grabbed both her hands[[ and]] squeezed them,
- +16.5 `loving-t08-s5`:  FOR SAVING OUR HOUSE," it said in shaky capitals.[[ I]] pressed my hand

### proud: tokens where MLP-25 writes most along `modes`

- +34.6 `loving-t05-s5`: . He put the tin in my basket and pulled me into[[ a]] hug with one
- +32.4 `happy-t08-s5`: , thank you, and I think I reached across the desk[[ and]] shook his hand
- +28.4 `adult-happy-s4`:  of them moved. Then she dropped the backpack on the tile[[ and]] ran at him
- +28.2 `calm-t12-s5`:  We crossed the line side by side, and she squeezed my[[ arm]], and we
- +28.2 `proud-t18-s5`:  them, her eyes shining. At the end the council member[[ shook]] his hand firmly
- +28.2 `loving-t05-s5`:  said. He put the tin in my basket and pulled me[[ into]] a hug with
- +27.8 `calm-t12-s5`: . We crossed the line side by side, and she squeezed[[ my]] arm, and
- +27.0 `sad-t05-s5`: ana looked up and her eyes filled immediately. They held each[[ other]] right there by
- +26.7 `loving-t05-s5`:  I said. He put the tin in my basket and pulled[[ me]] into a hug
- +26.3 `calm-t12-s5`:  line side by side, and she squeezed my arm, and[[ we]] went off to
- +26.1 `loving-t05-s5`:  and pulled me into a hug with one arm, thumping[[ my]] back. "
- +26.1 `adult-happy-s4`:  the tile and ran at him so hard he nearly fell over[[,]] and he heard
- +26.0 `adult-happy-s1`:  looked over. The happiness is hard to describe. My wife[[ cried]] a little when
- +25.6 `happy-t07-s5`:  it in person tomorrow and he said, "Seriously?" with[[ such]] an eager look
- +25.5 `adult-happy-s4`: . Then she dropped the backpack on the tile and ran at[[ him]] so hard he

### proud: tokens where MLP-25 writes most along `remaining`

- +13.2 `proud-t13-s5`:  Bongani crossed his arms and leaned against the fence and[[ said]] nothing at all
- +12.5 `happy-t04-s5`:  to my friend Oluwaseun and he said it[[ was]] the shini
- +12.3 `happy-t04-s5`:  saw Mum waiting I held the paper up over my head and[[ shouted]], and she
- +12.2 `happy-t04-s5`:  showed it to my friend Oluwaseun and he[[ said]] it was the
- +12.0 `adult-loving-s4`:  like a lighthouse and handed it over with the seriousness of[[ a]] jeweler.
- +12.0 `happy-t15-s5`:  and I showed her, turning around to model it like it[[ was]] haute couture
- +11.9 `happy-t18-s5`:  book to his chest and spun around to show his mom,[[ yelling]], "M
- +11.9 `adult-proud-s4`: . The science fair had selected her project on coastal erosion for[[ the]] state round.
- +11.8 `happy-t05-s5`:  she squeaked. Beatrix nodded, eyes sparkling, and[[ said]] she'd only
- +11.8 `inspired-t18-s5`:  sheepishly. She studied it for a long time and then[[ said]], "Can
- +11.8 `proud-t10-s5`: Chef Annick Moreau had cooked for prime ministers,[[ but]] tonight she was
- +11.8 `proud-t08-s5`:  leave, she paused at the door, turned back, and[[ said]], "You
- +11.7 `happy-t07-s5`:  see it in person tomorrow and he said, "Seriously?"[[ with]] such an eager
- +11.6 `proud-t18-s5`: . At the end the council member shook his hand firmly and[[ said]], "You
- +11.6 `adult-loving-s1`:  him I loved him for no reason at all, and he[[ said]], "I

### proud: example texts (bold: the top 5% of all tokens for that direction)

**proud-t00-s5**
- `main` (cut +7.0): Desmond was nine, and he had painted forty-two pickets.**He**(+8.0)  knew because he had**counted**(+7.6) **them**(+7.3)  twice**,**(+8.6) **once**(+8.1)  on Saturday and once on Sunday morning before anyone else was awake. The neighbor, Mrs. Abernathy, had wanted a fence between her vegetable garden and his family's yard to keep their beagle out of the tomatoes, and his mother had volunteered the two of them to help. Now Mrs. Abernathy stood at the end of the line in her sun hat, inspecting. Desmond stood**very**(+9.5)  straight beside her**with**(+10.0) **his**(+7.3)  hands behind his back**,**(+9.1)  white paint drying on his knuckles and one streak across his forehead. She**walked**(+7.3)  slowly. She bent to look at the bottom
- `modes` (cut +5.9): Desmond was nine, and he had painted forty-two pickets.**He**(+6.8)  knew because he had counted**them**(+5.9)  twice**,**(+8.1) **once**(+8.1) **on**(+6.8) **Saturday**(+6.4)  and**once**(+6.1)  on Sunday morning before anyone else was awake. The neighbor, Mrs. Abernathy, had wanted a fence between her vegetable garden and his family's yard to keep their beagle out of the tomatoes, and his mother had volunteered the two of them to help. Now Mrs. Abernathy**stood**(+7.0)  at the end of the line in her sun hat, inspecting. Desmond**stood**(+7.0) **very**(+9.8)  straight beside her**with**(+10.1) **his**(+6.5)  hands behind his back**,**(+6.2)  white paint drying on his knuckles and one streak across his forehead.**She**(+6.4)  walked slowly. She**bent**(+7.7)  to look at the bottom
- `remaining` (cut +6.1): Desmond was nine, and he had painted forty-two pickets.**He**(+6.3)  knew because he had**counted**(+7.9)  them twice**,**(+6.2)  once on Saturday and once on Sunday morning before anyone else was awake. The neighbor, Mrs. Abernathy, had wanted a fence between her vegetable garden and his family's yard to keep their beagle out of the tomatoes, and his mother had volunteered**the**(+6.4)  two of them to help. Now Mrs. Abernathy stood at the end of the line in her sun hat, inspecting. Desmond stood**very**(+6.4)  straight beside her**with**(+6.8)  his hands behind his back**,**(+8.1)  white paint drying on his knuckles and one streak across his forehead. She**walked**(+6.2)  slowly. She bent to look at the bottom

**proud-t01-s5**
- `main` (cut +7.0): Yusuf had never received a letter addressed to him alone before. At nineteen, after two years of sharing a room with his cousins, this was his first apartment, his name alone on the lease,**and**(+8.3)  the landlord's letter confirmed the move-in inspection: unit returned in excellent condition, full deposit to be applied forward. He had scrubbed the grout**with**(+7.3)  a toothbrush. He had patched two nail holes**with**(+8.7)  spackle he'd**learned**(+7.9)  to**use**(+8.0)  from**a**(+7.0)  video. He**stood**(+8.7)  in the doorway of the tiny studio holding the letter**and**(+8.4)  looked**around**(+8.1)  at the clean floor and the bare white walls**as**(+8.4)  if they**were**(+9.0) **a**(+11.1) **kingdom**(+8.6) . Then he called his mother. He tried to**sound**(+9.5)  offhand, telling
- `modes` (cut +5.9): Yusuf had never received a letter addressed to him alone before. At nineteen, after two years of sharing a room with his cousins, this was his first apartment, his name alone on the lease, and the landlord's letter confirmed the move-in inspection: unit returned in excellent condition, full deposit to be applied forward.**He**(+7.4)  had scrubbed the grout with a toothbrush. He had patched two nail holes**with**(+7.5)  spackle he'd learned to**use**(+7.7)  from**a**(+6.3)  video. He**stood**(+11.8)  in the doorway of the tiny studio holding the**letter**(+6.0) **and**(+7.7)  looked**around**(+8.4)  at the**clean**(+6.0)  floor and the bare white walls**as**(+6.0)  if they were a kingdom. Then he called his mother. He tried to sound offhand, telling
- `remaining` (cut +6.1): Yusuf had never received a letter addressed to him alone before. At nineteen, after two years of sharing a room with his cousins, this was his first apartment, his name alone on the lease,**and**(+8.1)  the landlord's letter confirmed the move-in inspection: unit returned in excellent condition, full deposit to be applied forward. He had scrubbed the grout with a toothbrush. He had patched two nail holes**with**(+6.7)  spackle**he**(+6.6) **'d**(+6.7) **learned**(+8.0)  to use from a video. He stood in the doorway**of**(+6.5)  the tiny studio holding the letter**and**(+6.2)  looked around at the clean floor and the bare white walls**as**(+7.3)  if they**were**(+9.2) **a**(+11.0) **kingdom**(+8.2) . Then he called his mother. He tried to**sound**(+9.0)  offhand, telling

**proud-t02-s5**
- `main` (cut +7.0): Aaliyah had organized the entire relief shipment herself: four hundred kilos of medical supplies, a borrowed cargo hold, and seventeen volunteers who were now sitting around Gate 41 waiting out a two-hour delay. She moved among them**with**(+9.0) **a**(+7.6)  clipboard**she**(+7.2)  didn't need, checking names she had already checked, because it gave her an excuse to look at them all. Six months ago this had been a conversation over chai with a doctor friend. Now there were matching blue T-shirts and labeled crates and a news reporter who had called**her**(+7.4)  that morning for a quote. When one of the younger volunteers**asked**(+7.5)  how she had pulled it all together**,**(+9.0)  Aali**yah**(+8.7) **laughed**(+7.7) **and**(+11.0)  waved the question off, but**her**(+9.1) 
- `modes` (cut +5.9): Aaliyah had organized the entire relief shipment herself: four hundred kilos of medical supplies, a borrowed cargo hold, and seventeen volunteers who were now sitting around Gate 41 waiting out a two-hour delay. She moved among them**with**(+6.1) **a**(+6.2)  clipboard**she**(+6.0)  didn't need, checking names she had already checked, because it gave her an excuse to look at them all. Six months ago this had been a conversation over chai with a doctor friend. Now there were matching blue T-shirts and labeled crates and a news reporter who had called**her**(+6.1)  that morning for a quote. When one of the younger volunteers asked how she had pulled it all together**,**(+7.6)  Aali**yah**(+7.7) **laughed**(+6.5) **and**(+10.0)  waved the question off, but**her**(+7.7) 
- `remaining` (cut +6.1): Aaliyah had organized the entire relief shipment herself: four hundred kilos of medical supplies, a borrowed cargo hold, and seventeen volunteers who were now sitting around Gate 41 waiting out a two-hour delay. She moved among them**with**(+7.9)  a clipboard she didn't need, checking names she had already checked, because it gave her an excuse to look at them all. Six months ago this had been a conversation over chai with a doctor friend. Now there were matching blue T-shirts and labeled crates and a news reporter who had called her that morning for a quote. When one of the younger volunteers**asked**(+6.5)  how she had pulled it all together**,**(+7.1)  Aali**yah**(+6.7)  laughed**and**(+8.3)  waved the question off, but**her**(+7.2) 
