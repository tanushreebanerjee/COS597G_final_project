### For random testing purposes

import numpy as np

context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."

sentences = context.split(".")

idx = np.random.choice(len(sentences), 1)[0]

print(idx)

repeat_sent = sentences[idx]

repeat_times = 3

final_context = sentences[:idx] + [repeat_sent] * repeat_times + sentences[idx+1:]

print(".".join(final_context))