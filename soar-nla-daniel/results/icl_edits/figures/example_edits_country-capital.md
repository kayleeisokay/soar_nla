# Example explanation edits: country-capital item 0

Base query 'China' (answer 'Beijing') -> target query 'Chile' (answer 'Santiago'). Raw-text prompt, layer-32 NLA; the explanation is the AV's account of the final prompt token (`:`).

## Seed 0

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Beijing' | 'Lima' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Beijing" is a standard geographic label, requiring the name of China's capital to complete the answer — "Beijing" is the city name for China.

Final token "are:" ends a location list ("the Republic and..."), requiring a city name or capitalized country label — almost certainly "Beijing" or "Beijing, China" — completing the "Capital:..." answer format, likely followed by "Beijing" or "Beijing, the capital." or "Shanghai" to name the capital city of China's government.
```

### Query-and-answer substitution

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Santiago" is a standard geographic label, requiring the name of Chile's capital to complete the answer — "Santiago" is the city name for Chile.

Final token "are:" ends a location list ("the Republic and..."), requiring a city name or capitalized country label — almost certainly "Santiago" or "Santiago, Chile" — completing the "Capital:..." answer format, likely followed by "Santiago" or "Santiago, the capital." or "Shanghai" to name the capital city of Chile's government.
```

### Final-paragraph rewrite

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Beijing" is a standard geographic label, requiring the name of China's capital to complete the answer — "Beijing" is the city name for China.

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Santiago" is a standard geographic label, requiring the name of Chile's capital to complete the answer — "Santiago" is the city name for Chile.

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 1

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Santiago' | 'Lima' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured geography/country fact-sheet format: the answer follows a list of factual entries, with "Capital: Beijing" as a location label.

The phrase "Country name:" signals a standard answer pattern — "China" is being named as the capital city, following the pattern "Capital: Beijing."

Final token "are:" ends a city name citation ("the capital: ...and..."), immediately requiring a geographic term like "Beijing" to complete the answer, likely "Beijing" or "Shanghai" or "Beijing, China" — the capital city name paired with "The capital is the" or "Chengdu." or "Beijing" as a list/city pairing.
```

### Query-and-answer substitution

```
Structured geography/country fact-sheet format: the answer follows a list of factual entries, with "Capital: Santiago" as a location label.

The phrase "Country name:" signals a standard answer pattern — "Chile" is being named as the capital city, following the pattern "Capital: Santiago."

Final token "are:" ends a city name citation ("the capital: ...and..."), immediately requiring a geographic term like "Santiago" to complete the answer, likely "Santiago" or "Shanghai" or "Santiago, Chile" — the capital city name paired with "The capital is the" or "Chengdu." or "Santiago" as a list/city pairing.
```

### Final-paragraph rewrite

```
Structured geography/country fact-sheet format: the answer follows a list of factual entries, with "Capital: Beijing" as a location label.

The phrase "Country name:" signals a standard answer pattern — "China" is being named as the capital city, following the pattern "Capital: Beijing."

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured geography/country fact-sheet format: the answer follows a list of factual entries, with "Capital: Santiago" as a location label.

The phrase "Country name:" signals a standard answer pattern — "Chile" is being named as the capital city, following the pattern "Capital: Santiago."

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 2

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Santiago' | 'Santiago' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured list format with factual geographic/demographic information — encyclopedic style for a country profile.

The phrase "Capital: Beijing" establishes a pattern of listing national details for China, requiring a location name to complete the answer about the capital city.

Final token ":" follows "and the capital is:" — a standard geographic citation phrase, requiring the name "Beijing" immediately next, likely followed by "Beijing" or "Beijing, China" as the capital city. Then likely "Shanghai" or "Beijing" to name the capital city, or "Beijing" to represent the Chinese capital as the capital.
```

### Query-and-answer substitution

```
Structured list format with factual geographic/demographic information — encyclopedic style for a country profile.

The phrase "Capital: Santiago" establishes a pattern of listing national details for Chile, requiring a location name to complete the answer about the capital city.

Final token ":" follows "and the capital is:" — a standard geographic citation phrase, requiring the name "Santiago" immediately next, likely followed by "Santiago" or "Santiago, Chile" as the capital city. Then likely "Shanghai" or "Santiago" to name the capital city, or "Santiago" to represent the Chinese capital as the capital.
```

### Final-paragraph rewrite

```
Structured list format with factual geographic/demographic information — encyclopedic style for a country profile.

The phrase "Capital: Beijing" establishes a pattern of listing national details for China, requiring a location name to complete the answer about the capital city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured list format with factual geographic/demographic information — encyclopedic style for a country profile.

The phrase "Capital: Santiago" establishes a pattern of listing national details for Chile, requiring a location name to complete the answer about the capital city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 3

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Lima' | 'Lima' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured list format of country facts, with geographic and demographic information being presented in a factual, encyclopedic style.

The phrase "Capital: Beijing" establishes a pattern of naming a capital city, strongly implying the answer will be "Beijing" to complete the location pairing with China.

Final token ":" ends a geographic label ("the capital:…and the city is…"), requiring a city name as the answer — almost certainly "Beijing" or "Beijing, China" — followed by "Shanghai" or "Beijing" as the capital pair. Or "Chengdu" or "Beijing" to match the country name as the most populous city.
```

### Query-and-answer substitution

```
Structured list format of country facts, with geographic and demographic information being presented in a factual, encyclopedic style.

The phrase "Capital: Santiago" establishes a pattern of naming a capital city, strongly implying the answer will be "Santiago" to complete the location pairing with Chile.

Final token ":" ends a geographic label ("the capital:…and the city is…"), requiring a city name as the answer — almost certainly "Santiago" or "Santiago, Chile" — followed by "Shanghai" or "Santiago" as the capital pair. Or "Chengdu" or "Santiago" to match the country name as the most populous city.
```

### Final-paragraph rewrite

```
Structured list format of country facts, with geographic and demographic information being presented in a factual, encyclopedic style.

The phrase "Capital: Beijing" establishes a pattern of naming a capital city, strongly implying the answer will be "Beijing" to complete the location pairing with China.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured list format of country facts, with geographic and demographic information being presented in a factual, encyclopedic style.

The phrase "Capital: Santiago" establishes a pattern of naming a capital city, strongly implying the answer will be "Santiago" to complete the location pairing with Chile.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 4

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Beijing' | 'Santiago' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured list format with factual geographic/national data, requiring a factual answer about China's capital city.

The phrase "Capital: Beijing..." signals a standard country profile template, with "China, Hong Kong and Beijing" as the final answer to the city of the capital.

Final token ":" follows "Beijing, the capital: " — a location label requiring a city name as the answer, specifically "Beijing" as the capital, likely followed by "Beijing" or "Shanghai" to name the capital city. or "Beijing, the capital of China" or "Beijing" — an enumerated list of cities ("Shanghai") matching the Chinese political system.
```

### Query-and-answer substitution

```
Structured list format with factual geographic/national data, requiring a factual answer about Chile's capital city.

The phrase "Capital: Santiago..." signals a standard country profile template, with "Chile, Hong Kong and Santiago" as the final answer to the city of the capital.

Final token ":" follows "Santiago, the capital: " — a location label requiring a city name as the answer, specifically "Santiago" as the capital, likely followed by "Santiago" or "Shanghai" to name the capital city. or "Santiago, the capital of Chile" or "Santiago" — an enumerated list of cities ("Shanghai") matching the Chinese political system.
```

### Final-paragraph rewrite

```
Structured list format with factual geographic/national data, requiring a factual answer about China's capital city.

The phrase "Capital: Beijing..." signals a standard country profile template, with "China, Hong Kong and Beijing" as the final answer to the city of the capital.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured list format with factual geographic/national data, requiring a factual answer about Chile's capital city.

The phrase "Capital: Santiago..." signals a standard country profile template, with "Chile, Hong Kong and Santiago" as the final answer to the city of the capital.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 5

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Santiago' | 'Beijing' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
List format with structured factual information about a country, following geographic/encyclopedic style with bullet points.

The phrase "Capital:Beijing" establishes a geographic fact pattern — "China" is the answer to a country name, with "Beijing is the capital" signaling a standard location label.

Final token ":" ends a "Capital:" field label following "China" — immediately requires a city name or capital name value, specifically "Beijing" as the answer, likely followed by "Beijing" or "Shanghai" to complete the pairing. Or "Beijing" or "Beijing, China" or "Beijing, the capital of China" to name the city.
```

### Query-and-answer substitution

```
List format with structured factual information about a country, following geographic/encyclopedic style with bullet points.

The phrase "Capital:Santiago" establishes a geographic fact pattern — "Chile" is the answer to a country name, with "Santiago is the capital" signaling a standard location label.

Final token ":" ends a "Capital:" field label following "Chile" — immediately requires a city name or capital name value, specifically "Santiago" as the answer, likely followed by "Santiago" or "Shanghai" to complete the pairing. Or "Santiago" or "Santiago, Chile" or "Santiago, the capital of Chile" to name the city.
```

### Final-paragraph rewrite

```
List format with structured factual information about a country, following geographic/encyclopedic style with bullet points.

The phrase "Capital:Beijing" establishes a geographic fact pattern — "China" is the answer to a country name, with "Beijing is the capital" signaling a standard location label.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
List format with structured factual information about a country, following geographic/encyclopedic style with bullet points.

The phrase "Capital:Santiago" establishes a geographic fact pattern — "Chile" is the answer to a country name, with "Santiago is the capital" signaling a standard location label.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 6

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Santiago' | 'Santiago' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured geographic/travel reference format: factual, encyclopedic style with trivia entries listing countries and capitals.

The phrase "Capital: Beijing" establishes a list of country facts, with "China" as the country name requiring a standard answer for capital city.

Final token ":" follows "Beijing, the capital: " — a location label immediately demands a name value, almost certainly "Beijing" to complete the "capital, known as the" format. Then likely "Beijing" or "Shanghai" or "Beijing, China" matching the Chinese capital, or "Beijing" to name the capital of the People's Republic of China.
```

### Query-and-answer substitution

```
Structured geographic/travel reference format: factual, encyclopedic style with trivia entries listing countries and capitals.

The phrase "Capital: Santiago" establishes a list of country facts, with "Chile" as the country name requiring a standard answer for capital city.

Final token ":" follows "Santiago, the capital: " — a location label immediately demands a name value, almost certainly "Santiago" to complete the "capital, known as the" format. Then likely "Santiago" or "Shanghai" or "Santiago, Chile" matching the Chinese capital, or "Santiago" to name the capital of the People's Republic of Chile.
```

### Final-paragraph rewrite

```
Structured geographic/travel reference format: factual, encyclopedic style with trivia entries listing countries and capitals.

The phrase "Capital: Beijing" establishes a list of country facts, with "China" as the country name requiring a standard answer for capital city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured geographic/travel reference format: factual, encyclopedic style with trivia entries listing countries and capitals.

The phrase "Capital: Santiago" establishes a list of country facts, with "Chile" as the country name requiring a standard answer for capital city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 7

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Beijing' | 'Beijing' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured geographic/demographic information pattern: answer format listing facts about a country, with factual trivia style responses.

The phrase "Capital: Beijing" establishes a list of country metadata, now naming the capital city "China" — the expected answer format "China, the capital is Beijing" points to a famous city.

Final token ":" follows "capital: " after the name "Republic of China," requiring a geographic or political answer — likely "Beijing" or "Shanghai" as the capital, or "Beijing" to complete the location, then "Beijing" or "Beijing, Beijing" or "Changsha" — a place name matching the Chinese capital pairing.
```

### Query-and-answer substitution

```
Structured geographic/demographic information pattern: answer format listing facts about a country, with factual trivia style responses.

The phrase "Capital: Santiago" establishes a list of country metadata, now naming the capital city "Chile" — the expected answer format "Chile, the capital is Santiago" points to a famous city.

Final token ":" follows "capital: " after the name "Republic of Chile," requiring a geographic or political answer — likely "Santiago" or "Shanghai" as the capital, or "Santiago" to complete the location, then "Santiago" or "Santiago, Santiago" or "Changsha" — a place name matching the Chinese capital pairing.
```

### Final-paragraph rewrite

```
Structured geographic/demographic information pattern: answer format listing facts about a country, with factual trivia style responses.

The phrase "Capital: Beijing" establishes a list of country metadata, now naming the capital city "China" — the expected answer format "China, the capital is Beijing" points to a famous city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured geographic/demographic information pattern: answer format listing facts about a country, with factual trivia style responses.

The phrase "Capital: Santiago" establishes a list of country metadata, now naming the capital city "Chile" — the expected answer format "Chile, the capital is Santiago" points to a famous city.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 8

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Beijing' | 'Beijing' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured geographic/encyclopedic format: factual Britannica-style article listing countries' capitals, with structured answer about China.

The phrase "capital is the China" signals a simple answer format — "City: Beijing" is a standard location citation for China's capital.

Final token ":" follows "the Republic of China and:" — this requires a location name as a capital city pair, specifically "Beijing" to match the "China" label, likely followed by "Beijing" or "Shanghai" or "Beijing, Beijing" or "Beijing; Shanghai" as a list of cities/capitals — the Chinese capital or the Beijing city name.
```

### Query-and-answer substitution

```
Structured geographic/encyclopedic format: factual Britannica-style article listing countries' capitals, with structured answer about Chile.

The phrase "capital is the Chile" signals a simple answer format — "City: Santiago" is a standard location citation for Chile's capital.

Final token ":" follows "the Republic of Chile and:" — this requires a location name as a capital city pair, specifically "Santiago" to match the "Chile" label, likely followed by "Santiago" or "Shanghai" or "Santiago, Santiago" or "Santiago; Shanghai" as a list of cities/capitals — the Chinese capital or the Santiago city name.
```

### Final-paragraph rewrite

```
Structured geographic/encyclopedic format: factual Britannica-style article listing countries' capitals, with structured answer about China.

The phrase "capital is the China" signals a simple answer format — "City: Beijing" is a standard location citation for China's capital.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured geographic/encyclopedic format: factual Britannica-style article listing countries' capitals, with structured answer about Chile.

The phrase "capital is the Chile" signals a simple answer format — "City: Santiago" is a standard location citation for Chile's capital.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

## Seed 9

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Santiago' | 'Santiago' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured list format with bullet points, requiring factual geographic/demographic information about a country.

The phrase "Capital: Beijing" establishes a pattern of listing country facts, with "China" as the final answer to "Location: United States of China" prompting a capital city citation.

Final token ":" follows "Beijing, the capital:" — this is a list of cities, requiring a categorical label like "Beijing" to name the capital, then likely "Beijing" or "Shanghai" as the capital city name, or "Beijing" continuing the "China/Beijing" pairing to match the Chinese city naming convention of "Beijing" or "Beijing..."
```

### Query-and-answer substitution

```
Structured list format with bullet points, requiring factual geographic/demographic information about a country.

The phrase "Capital: Santiago" establishes a pattern of listing country facts, with "Chile" as the final answer to "Location: United States of Chile" prompting a capital city citation.

Final token ":" follows "Santiago, the capital:" — this is a list of cities, requiring a categorical label like "Santiago" to name the capital, then likely "Santiago" or "Shanghai" as the capital city name, or "Santiago" continuing the "Chile/Santiago" pairing to match the Chinese city naming convention of "Santiago" or "Santiago..."
```

### Final-paragraph rewrite

```
Structured list format with bullet points, requiring factual geographic/demographic information about a country.

The phrase "Capital: Beijing" establishes a pattern of listing country facts, with "China" as the final answer to "Location: United States of China" prompting a capital city citation.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured list format with bullet points, requiring factual geographic/demographic information about a country.

The phrase "Capital: Santiago" establishes a pattern of listing country facts, with "Chile" as the final answer to "Location: United States of Chile" prompting a capital city citation.

Final token ":" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

