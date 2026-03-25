import pandas as pd
import numpy as np

np.random.seed(42)

SAMPLE_TRANSCRIPTS = {
    "AAPL": {
        (2024, 4): {
            "date": "2024-10-31",
            "content": """Operator: Good day, and welcome to Apple's Fourth Quarter Fiscal Year 2024 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Thank you. Good afternoon, everyone, and thanks for joining the call. We are pleased to report another strong quarter for Apple. Revenue came in at $94.9 billion, up 6% year over year, marking our best September quarter ever.

iPhone revenue was $46.2 billion, a new record for the September quarter. We are seeing incredible customer response to iPhone 16 and iPhone 16 Pro. The demand has been exceptionally strong, and we expect this momentum to continue through the holiday season.

Services revenue reached a new all-time high of $25 billion, growing 12% year over year. This is a testament to the strength of our ecosystem and the value our customers find in our services offerings.

Looking ahead, we are very optimistic about the holiday quarter. We anticipate revenue growth to accelerate, driven by the full quarter of iPhone 16 sales and continued momentum in Services. We expect total revenue to be between $123 billion and $127 billion.

However, we do face some headwinds in certain international markets due to foreign exchange pressures. The strong dollar may impact our reported results by approximately 2 percentage points. Additionally, the macroeconomic environment in China remains challenging, though we are cautiously optimistic about our position there.

Our gross margin was 46.2%, and we expect gross margins for the December quarter to be between 46% and 47%.

Luca Maestri -- Chief Financial Officer:
Thank you, Tim. Let me provide some additional details. Our operating expenses were well-managed at $14.3 billion. We returned over $29 billion to shareholders through dividends and share repurchases.

We generated $27 billion in operating cash flow, demonstrating the incredible cash generation capability of our business model.

Analyst Q&A:

Analyst: Tim, can you talk about what you're seeing in terms of upgrade rates for iPhone 16?

Tim Cook: We're very pleased with what we're seeing. The upgrade rates are strong, particularly for the Pro models. Apple Intelligence features are driving significant interest, and we believe this will be a multi-year upgrade cycle.

Analyst: Luca, are you seeing any margin pressure from the new product launches?

Luca Maestri: Our margins have been very resilient. We continue to optimize our supply chain, and the product mix has been favorable. We don't anticipate significant margin pressure going forward.

Analyst: How should we think about Services growth trajectory?

Tim Cook: Services is an increasingly important part of our business. We now have over 1 billion paid subscriptions. We expect double-digit growth to continue, driven by advertising, App Store, and our newer services like Apple TV+.
""",
        },
        (2024, 3): {
            "date": "2024-08-01",
            "content": """Operator: Good day, and welcome to Apple's Third Quarter Fiscal Year 2024 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Thank you, and good afternoon. We are reporting strong results for the June quarter, with revenue of $85.8 billion, up 5% year over year. This represents a return to growth after several quarters of year-over-year declines.

iPhone revenue was $39.3 billion. While this was slightly below some estimates, we are confident in the upcoming product cycle. We have exciting things planned that we believe will reinvigorate demand.

Services once again set a new all-time record at $24.2 billion, up 14% year over year. This is our strongest growth rate in Services in several quarters.

We are particularly excited about our AI initiatives. Apple Intelligence represents our vision for how AI should work - personal, private, and deeply integrated into the user experience. We anticipate this will drive a significant upgrade cycle beginning in the fall.

The Greater China region remains an area of uncertainty. Revenue was $14.7 billion, down 6% year over year. The competitive landscape there is intense, and macroeconomic conditions are challenging. We are taking a cautious approach to our outlook for that region.

Luca Maestri -- Chief Financial Officer:
Our gross margin was 46.3%, at the high end of our guidance range. Operating expenses were $14.3 billion. We expect September quarter revenue to grow at a similar rate to the June quarter, with gross margins between 45.5% and 46.5%.

Analyst Q&A:

Analyst: Tim, how do you see AI impacting iPhone demand?

Tim Cook: We believe Apple Intelligence is going to be transformative. The features we announced at WWDC have generated enormous interest. This could be the catalyst for the largest upgrade cycle we've seen in years.

Analyst: Can you quantify the China risk?

Tim Cook: China is always difficult to predict. We're seeing competitive dynamics that are unique to that market. However, we believe our ecosystem advantages and brand strength position us well for the long term.
""",
        },
        (2024, 2): {
            "date": "2024-05-02",
            "content": """Operator: Good day, and welcome to Apple's Second Quarter Fiscal Year 2024 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Good afternoon. Revenue for the March quarter was $90.8 billion. While this represents a slight decline of 4% year over year, it was better than our expectations going into the quarter.

iPhone revenue was $45.9 billion, down 10% year over year. This was impacted by a tough comparison to the prior year and some demand softness in certain markets. We are seeing some pressure in the smartphone market globally.

However, Services delivered another all-time record at $23.9 billion, growing 14% year over year. Our installed base of active devices has reached a new all-time high of 2.2 billion devices.

We are cautious about the near-term outlook. The macroeconomic environment remains uncertain, and consumer spending patterns could shift. We may see continued headwinds from foreign exchange.

iPad and Mac had a difficult quarter, with revenue declining year over year. We have exciting product refreshes planned that should help reinvigorate these categories.

Analyst Q&A:

Analyst: Tim, what gives you confidence in a return to growth?

Tim Cook: We have a robust product pipeline. The upcoming iPhone cycle, combined with our AI investments, gives us confidence in the second half. But I want to be measured in my outlook given the macro uncertainties.
""",
        },
        (2024, 1): {
            "date": "2024-02-01",
            "content": """Operator: Welcome to Apple's First Quarter Fiscal Year 2024 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Good afternoon, everyone. I'm thrilled to report record revenue of $119.6 billion for the December quarter, up 2% year over year. This is our highest quarterly revenue ever.

iPhone revenue was a record $69.7 billion. The iPhone 15 lineup has been incredibly well received, with particularly strong demand for iPhone 15 Pro and Pro Max.

Services set yet another all-time record at $23.1 billion, growing 11% year over year. Every geographic segment grew in Services.

We expect the March quarter to be roughly in line with the year-ago quarter. We are confident in our product roadmap and the fundamental strength of our ecosystem.

Greater China revenue was $20.8 billion, down 13% year over year. This is a challenging market, and we're facing headwinds from both macro conditions and competitive dynamics. We believe our differentiated ecosystem will ultimately prevail, but we should be realistic about near-term challenges.
""",
        },
        (2023, 4): {
            "date": "2023-11-02",
            "content": """Operator: Welcome to Apple's Fourth Quarter Fiscal Year 2023 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Good afternoon. Revenue for the September quarter was $89.5 billion, down 1% year over year. While we returned to growth in iPhone and Services, our overall results were impacted by continued declines in Mac and iPad.

iPhone revenue was $43.8 billion, up 3% year over year. The iPhone 15 launch has been very strong, with exceptional demand for the new titanium design and camera capabilities.

Services reached a new all-time record of $22.3 billion, growing 16% year over year. This is our strongest Services growth in six quarters.

Looking ahead, we face uncertainty in the macroeconomic environment. Consumer confidence has been volatile, and we're monitoring spending patterns closely. We may see some seasonal headwinds in the March quarter.

Luca Maestri -- Chief Financial Officer:
Gross margin was 45.2%, in line with our guidance. We returned $25 billion to shareholders this quarter. For the December quarter, we expect revenue growth in the low single digits.
""",
        },
        (2023, 3): {
            "date": "2023-08-03",
            "content": """Operator: Welcome to Apple's Third Quarter Fiscal Year 2023 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Revenue for the June quarter was $81.8 billion, down 1% year over year, but better than we had guided. This marks our third consecutive quarter of year-over-year decline, but the trend is clearly improving.

iPhone revenue was $39.6 billion, down 2%. The smartphone market remains soft globally, but we're outperforming the broader market.

Services hit a new all-time record of $21.2 billion, up 8% year over year. Our installed base continues to grow, and monetization rates are improving.

We are cautiously optimistic about the upcoming product cycle. The September quarter should see a return to growth, driven by new product launches. However, the macro environment remains challenging, and we could see continued foreign exchange headwinds.
""",
        },
        (2023, 2): {
            "date": "2023-05-04",
            "content": """Operator: Welcome to Apple's Second Quarter Fiscal Year 2023 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Revenue for the March quarter was $94.8 billion, down 3% year over year but ahead of expectations. We saw a challenging environment across multiple product categories.

iPhone revenue was $51.3 billion, down 2%. Mac revenue declined 31% due to tough comparisons with last year's M1 cycle. iPad was down 13%.

Services grew to $20.9 billion, up 5% year over year. While this is a slower growth rate than we'd like, it reflects the difficult comparison with COVID-era spending.

The outlook remains uncertain. We expect June quarter revenue to be similar year over year, subject to macro conditions. Foreign exchange continues to be a significant headwind.
""",
        },
        (2023, 1): {
            "date": "2023-02-02",
            "content": """Operator: Welcome to Apple's First Quarter Fiscal Year 2023 Earnings Conference Call.

Tim Cook -- Chief Executive Officer:
Revenue for the December quarter was $117.2 billion, down 5% year over year. This was a difficult quarter impacted by supply chain disruptions, foreign exchange headwinds, and a challenging macro environment.

iPhone revenue was $65.8 billion, down 8%. Supply constraints for iPhone 14 Pro models significantly impacted our results. We believe demand was well above supply throughout most of the quarter.

Despite these challenges, Services reached a new all-time record of $20.8 billion. Our ecosystem continues to strengthen.

We expect the March quarter to be challenging as well. The macro environment shows no signs of improving, and we face continued FX headwinds. We may see further pressure on consumer electronics spending.
""",
        },
    },
    "MSFT": {
        (2024, 4): {
            "date": "2024-10-22",
            "content": """Operator: Welcome to Microsoft's First Quarter Fiscal Year 2025 Earnings Conference Call.

Satya Nadella -- Chief Executive Officer:
Thank you. We had a strong start to fiscal 2025, with revenue of $65.6 billion, up 16% year over year. This was driven by continued strength in our cloud and AI businesses.

Microsoft Cloud revenue surpassed $38.9 billion, up 22%. Azure and other cloud services revenue grew 34%, with AI services contributing approximately 12 percentage points to Azure growth. This is a significant acceleration from last quarter.

We are seeing unprecedented demand for our AI capabilities. Copilot adoption is exceeding our expectations across enterprise customers. We now have over 100,000 organizations using Microsoft 365 Copilot, and usage is increasing rapidly.

Going forward, we expect Azure growth to accelerate further as we bring more AI capacity online. We are investing heavily in data center infrastructure to meet this demand. Capital expenditures were $20 billion this quarter, and we anticipate continued elevated spending.

However, we want to be transparent about the margin implications of these investments. While we expect operating margins to expand over time as AI revenue scales, near-term margins may face pressure from infrastructure buildout costs.

Amy Hood -- Chief Financial Officer:
For the December quarter, we expect revenue between $68.1 billion and $69.1 billion. We expect Azure growth of 31-32% in constant currency. Operating margins should be approximately 45%.

Analyst Q&A:

Analyst: Satya, how do you think about AI monetization over the next 12-18 months?

Satya Nadella: We're at the beginning of a massive platform shift. The monetization opportunity is enormous - every layer of our stack, from infrastructure to applications, is being enhanced by AI. We expect AI to be a multi-hundred-billion-dollar revenue opportunity for Microsoft over time.

Analyst: Amy, can you walk us through the margin trajectory given the heavy investment?

Amy Hood: We expect operating margins to compress slightly in the near term as we invest in capacity. However, we expect to see margin expansion as we move through fiscal 2025, driven by AI revenue scaling and operating leverage.
""",
        },
        (2024, 3): {
            "date": "2024-07-30",
            "content": """Operator: Welcome to Microsoft's Fourth Quarter Fiscal Year 2024 Earnings Conference Call.

Satya Nadella -- Chief Executive Officer:
We closed fiscal 2024 with strong momentum. Revenue for the June quarter was $64.7 billion, up 15% year over year. Full-year revenue exceeded $245 billion.

Azure grew 29%, with AI contributing 8 percentage points. The AI story is real and accelerating. We are seeing enterprises move from experimentation to production deployments at scale.

Microsoft 365 commercial revenue grew 12%. LinkedIn revenue grew 10%. Gaming revenue was up 44%, driven by the Activision acquisition.

We anticipate fiscal 2025 to be a year of AI-driven acceleration. We expect double-digit revenue growth to continue across our commercial businesses.

The competitive landscape in cloud is intense, but we believe our comprehensive AI platform gives us a significant advantage. We are cautious about making specific predictions given the rapid pace of change in AI, but we are very confident in our strategic position.
""",
        },
        (2024, 2): {
            "date": "2024-04-25",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue for the March quarter was $61.9 billion, up 17% year over year. This was a strong quarter across all segments.

Azure grew 31%, driven by AI demand. Copilot for Microsoft 365 is gaining traction, with significant enterprise deals signed this quarter.

We are investing aggressively in AI infrastructure. Capital expenditures were $14 billion, up significantly. We expect continued elevated investment as demand for AI compute exceeds supply.

There may be some near-term margin pressure from these investments, but we are confident in the long-term return profile.
""",
        },
        (2024, 1): {
            "date": "2024-01-30",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue was $62 billion, up 18% year over year. This exceeded our expectations across all three segments.

Azure grew 30%, with AI contributing 6 percentage points. The demand for AI services continues to grow rapidly.

We are seeing strong adoption of Copilot across our product portfolio. Enterprise interest is at unprecedented levels, and we expect this to accelerate through the year.

Margins were strong at 44%, reflecting operating leverage and disciplined cost management. We expect margins to remain healthy even as we increase AI investment.
""",
        },
        (2023, 4): {
            "date": "2023-10-24",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue was $56.5 billion, up 13% year over year. Microsoft Cloud revenue was $31.8 billion, growing 24%.

Azure grew 29% in constant currency. We're beginning to see AI contribute meaningfully to Azure growth, adding approximately 3 percentage points.

The enterprise environment remains somewhat uncertain, with customers being thoughtful about spending. However, AI is creating a new wave of demand that is partially offsetting macro headwinds.

We launched Microsoft 365 Copilot for enterprise customers this quarter. Early feedback has been very positive, and we expect rapid adoption through calendar 2024.
""",
        },
        (2023, 3): {
            "date": "2023-07-25",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue was $56.2 billion, up 8% year over year. This was ahead of expectations, driven by better-than-expected Azure performance.

Azure grew 26%, showing signs of stabilization after several quarters of deceleration. Enterprise customers are beginning to optimize their cloud spend, but new AI workloads are providing incremental demand.

We are cautiously optimistic about the second half. AI tailwinds should offset some of the macro uncertainty we're seeing in enterprise IT budgets.
""",
        },
        (2023, 2): {
            "date": "2023-04-25",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue was $52.9 billion, up 7% year over year. Azure grew 27%, slightly ahead of expectations.

The macro environment remains challenging, with enterprises carefully managing IT budgets. We're seeing longer sales cycles and more scrutiny on deals.

However, AI is creating unprecedented excitement. Our partnership with OpenAI and the integration of AI across our product stack positions us well for the next wave of growth.

There are risks in the near term from a potential recession and continued enterprise spending caution. We may see further deceleration before the AI-driven recovery takes hold.
""",
        },
        (2023, 1): {
            "date": "2023-01-24",
            "content": """Satya Nadella -- Chief Executive Officer:
Revenue was $52.7 billion, up 2% year over year. This was a slower quarter, reflecting macro headwinds across the technology sector.

Azure grew 31%, decelerating from 35% last quarter. Enterprise customers are optimizing cloud spend, and we expect this trend to continue for the next few quarters.

We announced a multi-billion dollar investment in OpenAI and the integration of AI capabilities into Azure. We believe this will be a major growth driver, but the financial impact will take several quarters to materialize.

The near-term outlook is uncertain. We expect continued pressure on PC-related revenue and cautious enterprise spending. However, we believe Microsoft is well-positioned for the AI era.
""",
        },
    },
    "JPM": {
        (2024, 4): {
            "date": "2024-10-11",
            "content": """Operator: Welcome to JPMorgan Chase's Third Quarter 2024 Earnings Conference Call.

Jamie Dimon -- Chairman and Chief Executive Officer:
We reported strong results this quarter, with net income of $12.9 billion and EPS of $4.37. Revenue was $43.3 billion, up 6% year over year.

Net interest income was $23.5 billion, modestly above expectations. We continue to benefit from higher rates, though we expect NII to normalize as the rate environment evolves. With the Fed beginning to cut rates, we anticipate some NII headwinds in coming quarters.

Investment banking fees were $2.4 billion, up 31% year over year. The M&A pipeline is building, and we're cautiously optimistic about a recovery in capital markets activity.

Credit quality remains solid, though we're seeing early signs of normalization in consumer credit. Net charge-offs were $2.1 billion, in line with expectations. We're closely monitoring consumer spending patterns and building reserves prudently.

The economic outlook is uncertain. While the labor market remains strong, there are risks from geopolitical tensions, potential recession, and the upcoming election. We're being cautious in our provisioning.

Jeremy Barnum -- Chief Financial Officer:
Our CET1 ratio was 15.3%, well above regulatory requirements. We repurchased $6 billion of stock this quarter.

For the fourth quarter, we expect NII of approximately $22.5 billion, reflecting the impact of rate cuts. Expenses should be approximately $22 billion.

Analyst Q&A:

Analyst: Jamie, how are you thinking about credit risk heading into 2025?

Jamie Dimon: Credit has been exceptional, but we're not naive. Delinquencies in credit card are normalizing to pre-COVID levels, which is healthy. But if we see a recession, credit losses could increase meaningfully. We're being conservative in our reserves.
""",
        },
        (2024, 3): {
            "date": "2024-07-12",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Net income was $13.1 billion. Revenue was $51 billion, up 20%, though this includes a $7.9 billion Visa gain.

The banking franchise continues to perform well. NII was $22.9 billion. Investment banking was strong, with fees up 50% year over year.

We expect the rate environment to be a headwind going forward. The yield curve dynamics are shifting, and we could see NII pressure if rates come down faster than expected.

Consumer spending remains healthy but is showing signs of cooling. The lower-income consumer is under the most pressure.
""",
        },
        (2024, 2): {
            "date": "2024-04-12",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
We reported net income of $13.4 billion. Revenue was $42.4 billion, up 8%.

NII was $23.1 billion, better than expected. Credit costs were $1.9 billion with net charge-offs of $2 billion.

The banking environment is constructive, but risks abound. Geopolitical uncertainty, persistent inflation, and potential rate changes could all impact our outlook. We may face challenges if the economic expansion slows more than expected.

We continue to invest in technology and our platform to drive long-term growth.
""",
        },
        (2024, 1): {
            "date": "2024-01-12",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Q4 net income was $9.3 billion, impacted by the FDIC special assessment of $2.9 billion. Excluding this, we had a very strong quarter.

Revenue was $39.9 billion. NII was $24.2 billion, at the high end of guidance. Investment banking showed signs of recovery.

The 2024 outlook is constructive but uncertain. We expect NII to moderate from 2023 peaks as rate dynamics shift. We guide NII of approximately $90 billion for 2024.

Credit quality is still strong but normalizing. We expect net charge-offs to increase modestly through 2024.
""",
        },
        (2023, 4): {
            "date": "2023-10-13",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Net income was $13.2 billion. Revenue was $40.7 billion, up 22% year over year, driven primarily by NII growth.

NII was $22.9 billion, benefiting significantly from higher rates. This level may represent a peak, and we expect some normalization going forward.

Investment banking remains muted. Capital markets activity has been subdued, and we don't see a meaningful recovery until 2024. The IPO market is particularly challenging.

Credit is performing well, but we're building reserves in anticipation of potential deterioration. The consumer remains resilient for now, but higher rates are beginning to impact mortgage and auto lending.
""",
        },
        (2023, 3): {
            "date": "2023-07-14",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Record quarterly net income of $14.5 billion, driven by the First Republic acquisition and strong NII.

Revenue was $42.4 billion. NII was $21.9 billion. The First Republic integration is proceeding well and exceeding our expectations.

However, the economic outlook is highly uncertain. We face risks from inflation, rate uncertainty, and geopolitical tensions. I remain cautious about the possibility of a hard landing.

Credit quality is still strong, but we may see deterioration if economic conditions worsen. We're maintaining elevated reserves.
""",
        },
        (2023, 2): {
            "date": "2023-04-14",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Net income was $12.6 billion. Revenue was $39.3 billion, up 25% driven by NII.

NII was $20.8 billion, well above expectations. The rate environment continues to benefit our deposit franchise.

The banking crisis earlier this quarter was contained, but it highlighted vulnerabilities in the system. We acquired First Republic Bank, which we believe will be significantly accretive.

We remain cautious on the economic outlook. Inflation is persistent, and the full impact of rate hikes has not yet been felt. A recession remains a possibility.
""",
        },
        (2023, 1): {
            "date": "2023-01-13",
            "content": """Jamie Dimon -- Chairman and Chief Executive Officer:
Net income was $11 billion. Revenue was $35.6 billion, up 18% year over year.

NII was $20.3 billion, up 48% year over year, reflecting the benefit of higher rates. This is a significant tailwind, but it may moderate as deposit costs increase.

The economy is currently resilient, but storm clouds are on the horizon. We're facing the most dangerous geopolitical environment since World War II, and I remain cautious about the potential for an economic downturn.

Investment banking revenue was weak, down 57% year over year. Capital markets are essentially closed for new issuance.
""",
        },
    },
    "NVDA": {
        (2024, 4): {
            "date": "2024-11-20",
            "content": """Operator: Welcome to NVIDIA's Third Quarter Fiscal 2025 Earnings Conference Call.

Jensen Huang -- President and Chief Executive Officer:
We delivered record revenue of $35.1 billion, up 94% year over year. Data Center revenue was $30.8 billion, up 112% year over year.

The demand for AI compute continues to be extraordinary. Hopper demand remains incredibly strong, and we are seeing tremendous excitement for Blackwell. We shipped our first Blackwell GPUs this quarter, and customer demand far exceeds our supply.

Every major cloud provider, consumer internet company, and enterprise is racing to build AI infrastructure. We expect this demand to continue for the foreseeable future.

Blackwell production is ramping, and we expect significant Blackwell revenue in Q4. Supply constraints remain, but we are working closely with our supply chain partners to scale as rapidly as possible.

Going forward, we anticipate continued exceptional growth. The AI infrastructure buildout is still in its early stages. We expect Q4 revenue of approximately $37.5 billion.

Colette Kress -- Chief Financial Officer:
Gross margins were 74.6%. We expect near-term gross margins to dip slightly to the low 70s as we ramp Blackwell production, before recovering to the mid-70s.

Analyst Q&A:

Analyst: Jensen, how do you see the demand trajectory for AI compute over the next few years?

Jensen Huang: We're at the beginning of a $1 trillion infrastructure transformation. Every data center in the world needs to be modernized for the AI era. The demand signals we're seeing are unlike anything in our 30-year history.

Analyst: Are you concerned about customers pulling back on AI spending?

Jensen Huang: Not at all. AI is delivering real ROI for our customers. Every dollar invested in NVIDIA AI infrastructure generates significant revenue for our customers. This is not speculative — it's driven by real applications and real demand.
""",
        },
        (2024, 3): {
            "date": "2024-08-28",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $30 billion, up 122% year over year. Data Center revenue was $26.3 billion, up 154%.

The Hopper architecture continues to see insatiable demand. We are supply-constrained, and expect to remain so through the Blackwell transition.

AI sovereign infrastructure is a new growth vector. Nations around the world are building AI capabilities, representing billions in incremental demand.

We expect Q3 revenue of approximately $32.5 billion, driven by strong Hopper demand and initial Blackwell shipments.
""",
        },
        (2024, 2): {
            "date": "2024-05-22",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $26 billion, up 262% year over year. This was driven almost entirely by Data Center, which grew 427% to $22.6 billion.

The demand environment is extraordinary. Generative AI and large language models are driving a fundamental shift in computing. Every industry is investing in AI infrastructure.

We are confident in sustained demand. The next generation Blackwell platform will deliver another major step function in performance and efficiency.

We expect Q2 revenue of approximately $28 billion.
""",
        },
        (2024, 1): {
            "date": "2024-02-21",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Record revenue of $22.1 billion, up 265% year over year. Data Center revenue was $18.4 billion, up 409%.

The AI revolution is in full swing. We're seeing demand from cloud providers, enterprises, sovereign nations, and startups. The breadth of demand is unprecedented.

Margins expanded to 76%, reflecting strong pricing power and operating leverage.

We anticipate Q1 fiscal 2025 revenue of approximately $24 billion, representing continued exceptional growth.
""",
        },
        (2023, 4): {
            "date": "2023-11-21",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $18.1 billion, up 206% year over year. Data Center reached $14.5 billion.

Demand continues to significantly exceed supply. We are working to increase production as quickly as possible.

Generative AI has triggered a global infrastructure buildout that we believe will last for years. The opportunity ahead is enormous.

We expect Q4 revenue of approximately $20 billion. We are confident in sustained growth as AI adoption accelerates across industries.
""",
        },
        (2023, 3): {
            "date": "2023-08-23",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $13.5 billion, up 101% year over year. Data Center was $10.3 billion, up 171%.

The generative AI boom is driving extraordinary demand for our products. Large language models require massive compute, and NVIDIA GPUs are the platform of choice.

We expect this demand to continue and accelerate. Q3 revenue guidance is approximately $16 billion, well above consensus expectations.

There is some risk from export restrictions to China, which could impact our revenue by $3-4 billion annually. However, demand from other regions more than compensates.
""",
        },
        (2023, 2): {
            "date": "2023-05-24",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $7.2 billion, up 19% year over year but up 19% sequentially, marking the beginning of our AI-driven acceleration.

Data Center revenue was $4.3 billion. We are seeing a surge in demand driven by generative AI and large language models. This is a tipping point for accelerated computing.

We expect Q2 revenue of approximately $11 billion, dramatically above current estimates. The AI wave is real and it's here.

Gaming revenue was $2.2 billion. While still recovering from the crypto hangover, we see improving trends.
""",
        },
        (2023, 1): {
            "date": "2023-02-22",
            "content": """Jensen Huang -- President and Chief Executive Officer:
Revenue was $6.1 billion, down 21% year over year. This was a challenging quarter impacted by weakness in gaming and the China export restrictions.

Data Center revenue was $3.6 billion, relatively flat. Enterprise AI adoption is growing, but we face headwinds from macro uncertainty and China restrictions.

Gaming revenue was $1.8 billion, down 46%, reflecting the post-crypto correction and inventory adjustments.

The near-term outlook is uncertain, but we see significant potential from generative AI. ChatGPT and similar technologies are driving enormous interest in AI compute. We believe this could drive a major inflection in our Data Center business.

We expect Q1 revenue of approximately $6.5 billion.
""",
        },
    },
    "XOM": {
        (2024, 4): {
            "date": "2024-11-01",
            "content": """Operator: Welcome to ExxonMobil's Third Quarter 2024 Earnings Conference Call.

Darren Woods -- Chairman and Chief Executive Officer:
We delivered strong results with earnings of $8.6 billion. Production was 4.6 million barrels of oil equivalent per day, up 24% year over year including the Pioneer acquisition.

The Pioneer integration is ahead of schedule and above plan. We've already identified over $2 billion in synergies and expect to achieve our full synergy target ahead of the original timeline.

Our Permian Basin production reached 1.4 million barrels per day, positioning us as the largest Permian producer. We expect to continue growing Permian production through the decade.

However, the commodity price environment is uncertain. Oil prices have been volatile, and we may face headwinds if global economic growth slows. OPEC+ dynamics add additional uncertainty.

We remain committed to our capital allocation framework: investing in high-return projects, maintaining and growing the dividend, and buying back shares. We repurchased $4.8 billion of shares this quarter.

Downstream margins have weakened from the exceptional levels of 2022-2023. We expect continued normalization in refining margins.

Analyst Q&A:

Analyst: How do you see the oil supply-demand balance evolving?

Darren Woods: Global oil demand continues to grow, driven by emerging markets. While there are risks from an economic slowdown, the structural supply-demand fundamentals remain constructive. We're planning for a range of scenarios.
""",
        },
        (2024, 3): {
            "date": "2024-07-26",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
Earnings were $9.2 billion. The Pioneer acquisition closed in May, transforming our production profile.

Upstream earnings were $7.1 billion, benefiting from higher oil prices and increased production. Chemical margins remain under pressure.

We expect capital spending of $23-25 billion annually through 2027 to fund our growth plans. The Permian, Guyana, and LNG are our three key growth engines.

Commodity price risks remain. A global recession could significantly impact oil prices and our earnings.
""",
        },
        (2024, 2): {
            "date": "2024-04-26",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
First quarter earnings were $8.2 billion. Production was 3.8 million barrels per day before the Pioneer close.

Upstream fundamentals are solid, with Brent averaging $83 per barrel. Downstream margins have moderated from peaks but remain above historical averages.

The Pioneer acquisition is expected to close in May and will be immediately accretive. We are confident in achieving our synergy targets.

We face risks from geopolitical uncertainty and potential demand weakness. However, our diversified portfolio and low-cost production base position us well across scenarios.
""",
        },
        (2024, 1): {
            "date": "2024-02-02",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
Fourth quarter earnings were $7.6 billion. Full-year 2023 earnings were $36 billion.

Production averaged 3.7 million barrels per day. Guyana continues to ramp and is exceeding expectations.

The pending Pioneer acquisition will create the largest position in the Permian Basin. We expect significant synergies and long-term value creation.

The outlook for 2024 is constructive but uncertain. Oil prices have been volatile, and we're planning for a range of scenarios. Our breakeven cost remains among the lowest in the industry.
""",
        },
        (2023, 4): {
            "date": "2023-10-27",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
Third quarter earnings were $9.1 billion. Production was 3.7 million barrels per day, our highest in five years.

We announced the acquisition of Pioneer Natural Resources for $60 billion, creating a Permian Basin powerhouse. This is a transformative deal that will drive growth for decades.

Upstream earnings remain strong, though below the exceptional levels of 2022. Oil prices have been supportive, averaging around $82 per barrel.

Risks include potential oil demand destruction from a recession, regulatory uncertainty, and OPEC+ supply decisions.
""",
        },
        (2023, 3): {
            "date": "2023-07-28",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
Second quarter earnings were $7.9 billion. While down from the record levels of 2022, these are strong results reflecting our operational excellence.

Oil prices have stabilized in the $75-85 range. We believe this is a supportive level for our investment program.

Downstream margins have normalized significantly from the exceptional 2022 levels. We expect margins to remain at or slightly above historical averages.

We're continuing to invest in our low-carbon solutions business, though returns in this area remain uncertain and dependent on policy support.
""",
        },
        (2023, 2): {
            "date": "2023-04-28",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
First quarter earnings were $11.4 billion. While commodity prices have come down, our results demonstrate the strength of our diversified portfolio.

Production was 3.6 million barrels per day. Guyana production continues to ramp ahead of schedule.

We're cautious about the near-term macro outlook. Banking sector stress and potential recession risks could impact commodity demand. However, OPEC+ production cuts provide a floor for prices.
""",
        },
        (2023, 1): {
            "date": "2023-01-31",
            "content": """Darren Woods -- Chairman and Chief Executive Officer:
Fourth quarter earnings were $12.8 billion, capping a record year with full-year earnings of $55.7 billion.

These results reflect an extraordinary commodity price environment. We recognize that prices at these levels may not be sustainable and are planning conservatively.

Production was 3.7 million barrels per day. Capital spending was $22.7 billion.

The 2023 outlook is more challenging. Oil prices have moderated, and we may see further pressure if recession fears materialize. Downstream margins are also likely to normalize from extreme levels.
""",
        },
    },
}

ADDITIONAL_COMPANIES = ["AMZN", "JNJ", "PG", "META", "UNH"]


def _generate_consensus_data() -> pd.DataFrame:
    np.random.seed(42)
    records = []

    base_eps = {
        "AAPL": 1.50, "MSFT": 2.80, "AMZN": 0.95, "JPM": 4.00,
        "JNJ": 2.60, "XOM": 2.20, "PG": 1.55, "NVDA": 0.60,
        "META": 3.50, "UNH": 6.20,
    }

    quarters = [
        (2023, 1), (2023, 2), (2023, 3), (2023, 4),
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    ]

    for symbol, base in base_eps.items():
        eps = base
        for year, quarter in quarters:
            growth = np.random.normal(0.03, 0.08)
            eps = eps * (1 + growth)

            estimate_before = eps + np.random.normal(0, eps * 0.05)
            revision_noise = np.random.normal(0, 0.04)
            estimate_after = estimate_before * (1 + revision_noise)

            actual_eps = eps + np.random.normal(0, eps * 0.03)

            records.append({
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "estimate_before_call": round(estimate_before, 2),
                "estimate_after_call": round(estimate_after, 2),
                "actual_eps": round(actual_eps, 2),
                "revision_pct": round(
                    (estimate_after - estimate_before) / abs(estimate_before)
                    if estimate_before != 0 else 0, 4
                ),
            })

    return pd.DataFrame(records)


def _generate_feature_data() -> pd.DataFrame:
    np.random.seed(42)
    records = []

    quarters = [
        (2023, 1), (2023, 2), (2023, 3), (2023, 4),
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    ]

    company_profiles = {
        "AAPL": {"base_sentiment": 0.15, "volatility": 0.08},
        "MSFT": {"base_sentiment": 0.25, "volatility": 0.06},
        "AMZN": {"base_sentiment": 0.10, "volatility": 0.12},
        "JPM": {"base_sentiment": 0.05, "volatility": 0.10},
        "JNJ": {"base_sentiment": 0.08, "volatility": 0.05},
        "XOM": {"base_sentiment": 0.00, "volatility": 0.15},
        "PG": {"base_sentiment": 0.12, "volatility": 0.04},
        "NVDA": {"base_sentiment": 0.35, "volatility": 0.10},
        "META": {"base_sentiment": 0.20, "volatility": 0.09},
        "UNH": {"base_sentiment": 0.10, "volatility": 0.06},
    }

    for symbol, profile in company_profiles.items():
        for year, quarter in quarters:
            sentiment_mean = profile["base_sentiment"] + np.random.normal(
                0, profile["volatility"]
            )
            sentiment_mean = np.clip(sentiment_mean, -0.5, 0.5)

            records.append({
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "sentiment_mean": round(sentiment_mean, 4),
                "sentiment_variance": round(
                    abs(np.random.normal(0.05, 0.02)), 4
                ),
                "pct_negative_sentences": round(
                    np.clip(0.25 - sentiment_mean * 0.3 + np.random.normal(0, 0.05), 0.05, 0.6), 4
                ),
                "sentiment_delta": round(np.random.normal(0, 0.08), 4),
                "hedging_score": round(
                    np.clip(np.random.normal(0.12, 0.04), 0.02, 0.35), 4
                ),
                "forward_looking_ratio": round(
                    np.clip(np.random.normal(0.35, 0.08), 0.1, 0.7), 4
                ),
                "guidance_specificity": round(
                    np.clip(np.random.normal(0.4, 0.1), 0.1, 0.8), 4
                ),
                "risk_term_frequency": round(
                    np.clip(np.random.normal(0.08, 0.03), 0.01, 0.25), 4
                ),
                "topic_shift_score": round(
                    abs(np.random.normal(0, 0.15)), 4
                ),
                "num_sentences": int(np.random.normal(180, 40)),
                "qa_sentiment_gap": round(np.random.normal(0, 0.06), 4),
            })

    return pd.DataFrame(records)


SAMPLE_CONSENSUS = _generate_consensus_data()
SAMPLE_FEATURES = _generate_feature_data()


def get_sample_transcript(symbol: str, year: int, quarter: int) -> dict | None:
    company_data = SAMPLE_TRANSCRIPTS.get(symbol, {})
    quarter_data = company_data.get((year, quarter))
    if quarter_data:
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "date": quarter_data["date"],
            "content": quarter_data["content"],
        }
    return None


def get_all_sample_transcripts() -> pd.DataFrame:
    records = []
    for symbol, quarters in SAMPLE_TRANSCRIPTS.items():
        for (year, quarter), data in quarters.items():
            records.append({
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "date": data["date"],
                "content": data["content"],
            })
    return pd.DataFrame(records)


def get_sample_consensus() -> pd.DataFrame:
    return SAMPLE_CONSENSUS.copy()


def get_sample_features() -> pd.DataFrame:
    return SAMPLE_FEATURES.copy()


def get_available_companies() -> list[str]:
    return list(SAMPLE_TRANSCRIPTS.keys()) + ADDITIONAL_COMPANIES


def get_available_quarters(symbol: str) -> list[tuple[int, int]]:
    if symbol in SAMPLE_TRANSCRIPTS:
        return sorted(SAMPLE_TRANSCRIPTS[symbol].keys(), reverse=True)
    return [
        (2024, 4), (2024, 3), (2024, 2), (2024, 1),
        (2023, 4), (2023, 3), (2023, 2), (2023, 1),
    ]
