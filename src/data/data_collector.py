
from dataset_builder import DatasetBuilder

if __name__ == "__main__":
    builder = DatasetBuilder()

    # ── CAT Quantitative Aptitude ────────────────────────────
    builder.add_text("""
    Time speed and distance problems are fundamental to CAT quantitative aptitude.
    If a train travels at 72 kilometres per hour its speed in metres per second is 20.
    Two trains start from stations A and B towards each other at 60 and 90 kilometres per hour.
    If the distance between A and B is 300 kilometres they will meet after 2 hours.
    A car covers 240 kilometres in 4 hours so its average speed is 60 kilometres per hour.
    Relative speed of two objects moving in the same direction is the difference of their speeds.
    Relative speed of two objects moving in opposite directions is the sum of their speeds.
    If a man walks at 5 kilometres per hour he reaches his destination 10 minutes late.
    If he walks at 6 kilometres per hour he reaches 5 minutes early.
    The distance to his destination can be calculated using the formula for time difference.
    Percentage profit is calculated as profit divided by cost price multiplied by 100.
    If an item is bought for 500 rupees and sold for 600 rupees the profit is 20 percent.
    Simple interest equals principal multiplied by rate multiplied by time divided by 100.
    Compound interest is calculated on the principal and the accumulated interest.
    The difference between compound interest and simple interest for 2 years is P times R squared divided by 10000.
    Ratio and proportion problems involve comparing two quantities by division.
    If A and B are in ratio 3 to 4 and B and C are in ratio 2 to 3 then A to C is 1 to 2.
    Mixture and alligation problems involve combining two mixtures of different concentrations.
    The alligation rule states that the ratio of quantities mixed equals the inverse ratio of differences from the mean.
    Work and time problems use the formula work equals rate multiplied by time.
    If A can complete a work in 10 days and B in 15 days together they complete it in 6 days.
    Pipes and cisterns follow the same principle as work and time.
    A pipe filling a tank in 4 hours has a rate of one quarter of the tank per hour.
    Number series problems require identifying the pattern in a sequence of numbers.
    Arithmetic progressions have a common difference between consecutive terms.
    Geometric progressions have a common ratio between consecutive terms.
    The sum of first n natural numbers is n multiplied by n plus 1 divided by 2.
    The sum of squares of first n natural numbers is n times n plus 1 times 2n plus 1 divided by 6.
    Permutations refer to arrangements where order matters.
    Combinations refer to selections where order does not matter.
    The number of ways to arrange n objects is n factorial.
    The number of ways to choose r objects from n is n choose r equals n factorial divided by r factorial times n minus r factorial.
    Probability is defined as favourable outcomes divided by total outcomes.
    The probability of an event and its complement always sum to 1.
    Venn diagrams are used to solve problems involving sets and their intersections.
    If two sets A and B have some common elements the union is A plus B minus the intersection.
    """, source="CAT_quant")

    builder.add_text("""
    Quadratic equations have two roots which can be real or complex.
    The discriminant of a quadratic equation is b squared minus 4ac.
    If the discriminant is positive the roots are real and distinct.
    If the discriminant is zero the roots are real and equal.
    If the discriminant is negative the roots are complex conjugates.
    Linear equations in two variables can be solved by substitution or elimination.
    The slope of a line passing through two points is rise over run.
    The equation of a line in slope intercept form is y equals mx plus c.
    A circle with centre at origin and radius r has equation x squared plus y squared equals r squared.
    The distance between two points in a coordinate plane uses the distance formula.
    Trigonometric ratios are defined for acute angles in a right triangle.
    Sine is opposite over hypotenuse, cosine is adjacent over hypotenuse.
    Tangent is opposite over adjacent and equals sine divided by cosine.
    The Pythagorean identity states sine squared plus cosine squared equals one.
    Heights and distances problems use trigonometric ratios and the angle of elevation.
    Mensuration involves calculating area perimeter and volume of geometric shapes.
    The area of a circle is pi times radius squared.
    The circumference of a circle is 2 times pi times radius.
    The volume of a sphere is four thirds times pi times radius cubed.
    The surface area of a sphere is 4 times pi times radius squared.
    The volume of a cylinder is pi times radius squared times height.
    Data interpretation requires reading tables bar charts line graphs and pie charts.
    A bar chart shows categorical data with rectangular bars proportional to values.
    A pie chart shows proportions of a whole as slices of a circle.
    Line graphs show trends over time with data points connected by lines.
    """, source="CAT_quant_2")

    # ── CAT Verbal Ability ───────────────────────────────────
    builder.add_text("""
    Reading comprehension is a critical section of the CAT verbal ability paper.
    A passage is followed by questions testing understanding inference and vocabulary.
    Parajumbles require rearranging sentences to form a coherent and logical paragraph.
    The opening sentence of a paragraph usually introduces the main idea or topic.
    Sentence correction tests knowledge of grammar syntax and usage.
    Subject verb agreement requires the verb to match the number of the subject.
    A singular subject takes a singular verb while a plural subject takes a plural verb.
    Pronouns must agree with their antecedents in number gender and person.
    Parallel structure requires similar grammatical forms for similar ideas.
    Active voice makes the subject the doer of the action.
    Passive voice makes the subject the receiver of the action.
    Critical reasoning involves identifying assumptions conclusions and logical flaws.
    An assumption is an unstated premise that the argument depends on.
    A conclusion is the main point the author is trying to establish.
    Strengthening an argument means providing additional support for the conclusion.
    Weakening an argument means providing evidence against the conclusion or premise.
    Vocabulary questions test knowledge of word meanings synonyms and antonyms.
    Etymology helps understand word meanings by tracing roots prefixes and suffixes.
    Contextual usage requires choosing the word that fits the meaning of a sentence.
    Fill in the blanks requires selecting the most appropriate word or phrase.
    Sentence completion tests the ability to understand the logical flow of ideas.
    Analogies test understanding of relationships between pairs of words.
    Idiomatic expressions have meanings different from their literal interpretation.
    Cloze passages require filling multiple blanks in a passage with appropriate words.
    """, source="CAT_verbal")

    builder.add_text("""
    The CAT exam tests verbal ability logical reasoning and quantitative aptitude.
    Verbal ability section includes reading comprehension parajumbles and sentence correction.
    Logical reasoning tests analytical thinking through puzzles and arrangements.
    Seating arrangement problems require placing people in positions based on given conditions.
    Blood relation problems involve determining family relationships from given information.
    Direction sense problems test the ability to track movement and orientation.
    Coding decoding problems involve finding patterns in how letters or numbers are transformed.
    Syllogism problems use logical deduction from given statements to reach conclusions.
    All A are B means every member of group A is also a member of group B.
    Some A are B means at least one member of A is also in B.
    No A are B means no member of A is a member of B.
    Data sufficiency problems require determining if given data is enough to solve a problem.
    Logical sequence problems test the ability to identify patterns in series of numbers letters or figures.
    Critical path analysis involves finding the longest sequence of dependent tasks.
    Decision making problems require applying given criteria to reach optimal decisions.
    """, source="CAT_reasoning")

    # ── UPSC General Studies Paper 1 ─────────────────────────
    builder.add_text("""
    The Indian National Congress was founded in 1885 by Allan Octavian Hume.
    The first session of the Indian National Congress was held in Bombay in December 1885.
    Dadabhai Naoroji was the first president of the Indian National Congress.
    The partition of Bengal in 1905 was announced by Lord Curzon the Viceroy of India.
    The partition of Bengal was revoked in 1911 during the Delhi Durbar.
    The Swadeshi Movement was launched in response to the partition of Bengal.
    The Muslim League was founded in Dhaka in 1906 by Aga Khan and others.
    The Morley Minto Reforms of 1909 introduced separate electorates for Muslims.
    The Lucknow Pact of 1916 was an agreement between the Congress and the Muslim League.
    The Home Rule Movement was launched by Bal Gangadhar Tilak and Annie Besant in 1916.
    The Rowlatt Act of 1919 allowed detention without trial and was widely opposed.
    The Jallianwala Bagh massacre took place on 13 April 1919 at Amritsar.
    General Dyer ordered troops to fire on a peaceful gathering killing hundreds.
    Rabindranath Tagore renounced his knighthood in protest against the massacre.
    The Non Cooperation Movement was launched by Mahatma Gandhi in 1920.
    The Khilafat Movement ran parallel to the Non Cooperation Movement.
    Gandhi called off the Non Cooperation Movement after the Chauri Chaura incident in 1922.
    The Simon Commission was appointed in 1927 to review the Indian constitution.
    The Simon Commission was boycotted because it had no Indian members.
    The Nehru Report of 1928 was the first attempt by Indians to draft a constitution.
    The Lahore Session of Congress in 1929 declared Purna Swaraj as its goal.
    Jawaharlal Nehru hoisted the tricolor flag on the banks of the Ravi on 31 December 1929.
    The Dandi March was undertaken by Gandhi from 12 March to 6 April 1930.
    Gandhi marched 241 miles from Sabarmati Ashram to Dandi to protest the salt tax.
    The Civil Disobedience Movement began with the Dandi March.
    The Gandhi Irwin Pact of 1931 suspended the Civil Disobedience Movement.
    The Round Table Conferences were held in London in 1930 1931 and 1932.
    The Government of India Act 1935 introduced provincial autonomy and a federal structure.
    The Quit India Movement was launched by Gandhi on 8 August 1942.
    The slogan Do or Die was given by Gandhi during the Quit India Movement.
    The Indian National Army was formed by Subhash Chandra Bose to fight for independence.
    Bose gave the slogan Give me blood and I will give you freedom.
    The Cabinet Mission Plan of 1946 proposed a federal structure with provinces and princely states.
    The partition of India led to the creation of India and Pakistan on 14 and 15 August 1947.
    Jawaharlal Nehru became the first Prime Minister of independent India.
    Dr Rajendra Prasad became the first President of the Republic of India.
    Dr B R Ambedkar was the Chairman of the Drafting Committee of the Indian Constitution.
    The Indian Constitution was adopted on 26 November 1949 and came into force on 26 January 1950.
    """, source="UPSC_modern_history")

    builder.add_text("""
    Ancient India had several great empires and civilizations.
    The Indus Valley Civilization flourished between 2500 and 1750 BCE.
    Major cities of the Indus Valley Civilization include Mohenjo daro and Harappa.
    The Indus Valley people had an advanced urban planning system with grid pattern streets.
    The Vedic Age began around 1500 BCE with the arrival of the Aryans.
    The Rigveda is the oldest of the four Vedas and contains hymns to gods.
    The Upanishads contain philosophical discussions about the nature of reality.
    The Mauryan Empire was founded by Chandragupta Maurya in 321 BCE.
    Chanakya also known as Kautilya was the chief minister and advisor to Chandragupta.
    The Arthashastra written by Kautilya is a treatise on statecraft economics and military strategy.
    Ashoka the Great was the most famous Mauryan emperor who ruled from 268 to 232 BCE.
    Ashoka converted to Buddhism after the Kalinga War around 261 BCE.
    The Gupta Empire is known as the Golden Age of India for its cultural and scientific achievements.
    Chandragupta I founded the Gupta Empire in 320 CE.
    Samudragupta expanded the empire and is called the Napoleon of India.
    Chandragupta II also known as Vikramaditya was the most powerful Gupta ruler.
    Aryabhata was a mathematician and astronomer of the Gupta period.
    Kalidasa was the greatest Sanskrit poet and playwright of the Gupta era.
    The Bhakti Movement emphasized personal devotion to God over rituals.
    Kabir was a poet saint of the Bhakti Movement who preached unity of all religions.
    Guru Nanak founded Sikhism in the 15th century in Punjab.
    The Delhi Sultanate was established by Qutb ud din Aibak in 1206 CE.
    Alauddin Khilji introduced market reforms and a price control system.
    The Mughal Empire was founded by Babur after the First Battle of Panipat in 1526.
    Akbar was the greatest Mughal emperor known for his religious tolerance and administration.
    Akbar introduced the Din i Ilahi a syncretic religion combining elements of Hinduism Islam Christianity and Zoroastrianism.
    Aurangzeb was the last powerful Mughal emperor known for his strict orthodox policies.
    The Maratha Empire rose under Chhatrapati Shivaji in the 17th century.
    """, source="UPSC_ancient_medieval")

    # ── UPSC General Studies Paper 2 — Polity ────────────────
    builder.add_text("""
    The Indian Constitution is the longest written constitution in the world.
    It originally had 395 articles 22 parts and 8 schedules.
    Currently the Constitution has 448 articles 25 parts and 12 schedules.
    The Preamble declares India to be a Sovereign Socialist Secular Democratic Republic.
    The words Socialist and Secular were added by the 42nd Constitutional Amendment in 1976.
    Fundamental Rights are guaranteed under Part 3 of the Indian Constitution from Articles 12 to 35.
    The six Fundamental Rights are Right to Equality Right to Freedom Right against Exploitation Right to Freedom of Religion Cultural and Educational Rights and Right to Constitutional Remedies.
    Article 14 guarantees equality before law and equal protection of laws.
    Article 19 guarantees six freedoms including freedom of speech and expression.
    Article 21 guarantees the right to life and personal liberty.
    Article 32 is the right to constitutional remedies described by Ambedkar as the heart and soul of the Constitution.
    Directive Principles of State Policy are contained in Part 4 from Articles 36 to 51.
    Directive Principles are non justiciable meaning they cannot be enforced by courts.
    Fundamental Duties were added by the 42nd Amendment and are contained in Article 51A.
    The Parliament of India consists of two houses Lok Sabha and Rajya Sabha.
    The Lok Sabha is the lower house or House of the People with maximum 552 members.
    The Rajya Sabha is the upper house or Council of States with maximum 250 members.
    The President of India is elected by an electoral college comprising elected members of Parliament and state legislatures.
    The President is the constitutional head of the executive.
    The Prime Minister is the head of the Council of Ministers and the real executive.
    The Council of Ministers is collectively responsible to the Lok Sabha.
    The Supreme Court of India is the apex court and final interpreter of the Constitution.
    The Chief Justice of India heads the Supreme Court.
    High Courts are the highest courts in each state.
    The Election Commission of India is an autonomous constitutional body that conducts elections.
    The Chief Election Commissioner cannot be removed except by impeachment like a Supreme Court judge.
    The Finance Commission is constituted every five years to recommend distribution of taxes between Union and States.
    The Planning Commission was replaced by NITI Aayog in 2015.
    NITI Aayog stands for National Institution for Transforming India.
    Panchayati Raj institutions were given constitutional status by the 73rd Amendment in 1992.
    The 74th Amendment gave constitutional status to urban local bodies or municipalities.
    The Governor is appointed by the President and is the constitutional head of the state.
    The Chief Minister is the head of the state council of ministers.
    Article 370 granted special autonomous status to Jammu and Kashmir.
    Article 370 was revoked in August 2019 and Jammu and Kashmir was bifurcated into two union territories.
    Emergency provisions are contained in Articles 352 360 and 356 of the Constitution.
    Article 352 relates to National Emergency which can be proclaimed on grounds of war external aggression or armed rebellion.
    Article 356 relates to President's Rule or State Emergency.
    Article 360 relates to Financial Emergency.
    """, source="UPSC_polity")

    builder.add_text("""
    The Union Executive consists of the President Vice President Prime Minister and Council of Ministers.
    The President of India can be removed by impeachment under Article 61.
    The Vice President is the ex officio Chairman of the Rajya Sabha.
    The Council of Ministers includes Cabinet Ministers Ministers of State and Deputy Ministers.
    Cabinet Ministers head important ministries and attend Cabinet meetings.
    Ministers of State may have independent charge of a ministry or assist a Cabinet Minister.
    The Attorney General of India is the chief law officer of the Government of India.
    The Comptroller and Auditor General audits the accounts of the Union and states.
    The Union Public Service Commission conducts civil services examinations.
    The Indian Administrative Service IAS and Indian Police Service IPS are All India Services.
    The Central Vigilance Commission is an independent statutory body overseeing corruption.
    The Central Information Commission oversees implementation of the Right to Information Act.
    The National Human Rights Commission protects and promotes human rights in India.
    Inter State Council is a constitutional body for coordination between centre and states.
    The Finance Minister presents the Union Budget in Parliament every year.
    The budget is presented on 1 February since 2017 instead of the last day of February.
    Money Bills can only be introduced in the Lok Sabha and Rajya Sabha cannot reject them.
    A Constitutional Amendment Bill can be introduced in either house of Parliament.
    The most important method of Constitutional amendment is under Article 368.
    Some provisions can be amended by simple majority while others require special majority.
    Federal features of the Indian Constitution include dual government dual citizenship and dual judiciary.
    Unitary features include a strong centre residual powers with centre and single citizenship.
    The Seventh Schedule contains three lists Union List State List and Concurrent List.
    The Union List has subjects on which only Parliament can make laws including defence and foreign affairs.
    The State List has subjects on which only state legislatures can make laws including police and public order.
    The Concurrent List has subjects on which both Parliament and states can legislate.
    In case of conflict between Union and State laws on Concurrent subjects Union law prevails.
    """, source="UPSC_polity_2")

    # ── UPSC Geography ───────────────────────────────────────
    builder.add_text("""
    India is located in South Asia and is the seventh largest country in the world by area.
    India covers an area of approximately 3.29 million square kilometres.
    India has a coastline of approximately 7516 kilometres.
    The Tropic of Cancer passes through eight Indian states.
    The Himalayan mountain range forms the northern boundary of India.
    The Himalayas are the world's highest mountain range with Mount Everest at 8848 metres.
    The Gangetic Plain is one of the most fertile agricultural regions in the world.
    The Deccan Plateau is a large triangular plateau in peninsular India.
    The Western Ghats receive heavy rainfall due to the southwest monsoon.
    The Eastern Ghats are discontinuous hills along the eastern coast of India.
    The Indian monsoon is driven by differential heating of land and sea.
    The southwest monsoon brings rainfall to most of India from June to September.
    The northeast monsoon brings rainfall to southeastern India from October to December.
    The Ganga is the longest river in India flowing approximately 2525 kilometres.
    The Brahmaputra enters India from Tibet through Arunachal Pradesh.
    The Krishna Godavari Kaveri and Mahanadi are important peninsular rivers.
    Peninsular rivers are rain fed and non perennial unlike Himalayan rivers which are perennial.
    India has diverse soil types including alluvial black red laterite and desert soils.
    Alluvial soil found in the Gangetic Plain is the most fertile and suitable for agriculture.
    Black soil also called regur soil is ideal for cotton cultivation in the Deccan.
    India has three main climate zones tropical monsoon subtropical and alpine.
    India has a diverse range of natural vegetation including tropical rainforests deciduous forests and grasslands.
    Project Tiger was launched in 1973 to protect the tiger population in India.
    India has over 50 tiger reserves as part of Project Tiger.
    The Sundarbans in West Bengal is the largest mangrove forest in the world.
    India is one of 17 megadiverse countries in the world.
    The Western Ghats is a biodiversity hotspot recognized by UNESCO.
    Coal iron ore bauxite and manganese are important mineral resources of India.
    Jharkhand Odisha and Chhattisgarh are mineral rich states of India.
    India is the world's largest producer and consumer of spices.
    """, source="UPSC_geography")

    # ── SSC General Knowledge ────────────────────────────────
    builder.add_text("""
    The national animal of India is the Bengal Tiger.
    The national bird of India is the Indian Peacock.
    The national flower of India is the Lotus.
    The national tree of India is the Banyan tree.
    The national river of India is the Ganga.
    The national fruit of India is the Mango.
    The national game of India is Field Hockey.
    The national emblem of India is the Lion Capital of Ashoka.
    The national anthem of India is Jana Gana Mana composed by Rabindranath Tagore.
    The national song of India is Vande Mataram composed by Bankim Chandra Chattopadhyay.
    The Reserve Bank of India was established on 1 April 1935.
    The Reserve Bank of India is the central bank of the country.
    The currency of India is the Indian Rupee with symbol Rs or INR.
    The Parliament of India is located in New Delhi.
    The Supreme Court of India is located in New Delhi.
    The capital of India is New Delhi.
    The largest state of India by area is Rajasthan.
    The smallest state of India by area is Goa.
    The most populous state of India is Uttar Pradesh.
    The largest union territory by area is Ladakh.
    The smallest union territory by population is Lakshadweep.
    India has 28 states and 8 union territories after the reorganization in 2019.
    The first Prime Minister of India was Jawaharlal Nehru who served from 1947 to 1964.
    The first President of India was Dr Rajendra Prasad who served from 1950 to 1962.
    The first woman Prime Minister of India was Indira Gandhi.
    The first woman President of India was Pratibha Patil.
    APJ Abdul Kalam was the 11th President of India and is known as the Missile Man.
    The Bharat Ratna is the highest civilian honour of India.
    Mother Teresa was the first non Indian to receive the Bharat Ratna in 1980.
    Sachin Tendulkar was the first sportsperson to receive the Bharat Ratna in 2014.
    The Nobel Prize is awarded annually in Physics Chemistry Medicine Literature Economics and Peace.
    Rabindranath Tagore was the first Indian to win the Nobel Prize in Literature in 1913.
    CV Raman won the Nobel Prize in Physics in 1930 for the Raman Effect.
    Amartya Sen won the Nobel Prize in Economics in 1998.
    Kailash Satyarthi shared the Nobel Peace Prize in 2014.
    The Olympic Games are held every four years.
    India has won several Olympic medals in field hockey wrestling shooting and athletics.
    The first Olympic Games of the modern era were held in Athens Greece in 1896.
    The FIFA World Cup is the biggest football tournament held every four years.
    The ICC Cricket World Cup is held every four years.
    India won the Cricket World Cup in 1983 and 2011.
    """, source="SSC_GK")

    builder.add_text("""
    Inventions and discoveries have shaped human civilization.
    The wheel was invented around 3500 BCE and revolutionized transportation.
    The printing press was invented by Johannes Gutenberg around 1440.
    The steam engine was invented by James Watt in 1769 revolutionizing industry.
    Alexander Graham Bell invented the telephone in 1876.
    Thomas Edison invented the phonograph in 1877 and the practical light bulb in 1879.
    The Wright brothers Orville and Wilbur made the first powered flight in 1903.
    Penicillin was discovered by Alexander Fleming in 1928.
    The structure of DNA was discovered by James Watson and Francis Crick in 1953.
    Tim Berners Lee invented the World Wide Web in 1989.
    Science and technology have brought tremendous progress to humanity.
    Newton's first law states that a body continues in its state of rest or motion unless acted upon by a force.
    Newton's second law states that force equals mass times acceleration.
    Newton's third law states that every action has an equal and opposite reaction.
    The law of gravitation states that every body attracts every other body with a force proportional to the product of their masses.
    Einstein's theory of relativity showed that energy equals mass times the speed of light squared.
    Archimedes principle states that a body immersed in a fluid experiences an upward buoyant force.
    Bernoulli's principle explains why aircraft can fly based on pressure differences.
    The periodic table organizes elements by atomic number and chemical properties.
    The atomic number of hydrogen is 1 helium is 2 carbon is 6 and oxygen is 8.
    Water is composed of two hydrogen atoms and one oxygen atom.
    Carbon dioxide consists of one carbon atom and two oxygen atoms.
    Photosynthesis is the process by which plants convert sunlight carbon dioxide and water into glucose.
    Respiration is the process by which organisms convert glucose into energy releasing carbon dioxide.
    The cell is the basic unit of life.
    DNA carries genetic information and is found in the nucleus of cells.
    Evolution by natural selection was proposed by Charles Darwin in 1859.
    Vaccines work by stimulating the immune system to produce antibodies against a disease.
    Antibiotics kill bacteria and have revolutionized the treatment of infectious diseases.
    """, source="SSC_science")

    # ── Economics ────────────────────────────────────────────
    builder.add_text("""
    Economics studies how individuals and societies allocate scarce resources.
    Microeconomics focuses on individual markets firms and consumers.
    Macroeconomics focuses on the economy as a whole including GDP inflation and unemployment.
    GDP or Gross Domestic Product measures the total value of goods and services produced in a country.
    India's GDP is one of the largest in the world making it a major emerging economy.
    Inflation is the rate at which the general level of prices is rising.
    The Reserve Bank of India uses monetary policy to control inflation.
    The repo rate is the rate at which the RBI lends to commercial banks.
    An increase in repo rate makes borrowing expensive thereby reducing money supply.
    The Consumer Price Index measures changes in the price level of consumer goods.
    The Wholesale Price Index measures prices at the producer level.
    Fiscal policy involves government decisions on taxation and spending.
    The Union Budget outlines the government's revenues and expenditures.
    Direct taxes include income tax and corporate tax paid directly by earners.
    Indirect taxes include GST which is collected by sellers and passed on to consumers.
    GST or Goods and Services Tax replaced multiple indirect taxes in India in 2017.
    Foreign Direct Investment or FDI refers to investments made by foreign entities in India.
    The Balance of Payments records all economic transactions between India and the rest of the world.
    The current account includes trade in goods services and transfer payments.
    The capital account includes investments and financial flows.
    The Human Development Index measures life expectancy education and income.
    India's rank on the Human Development Index has been improving steadily.
    Five Year Plans guided India's economic development from 1951 to 2017.
    NITI Aayog replaced the Planning Commission and formulates long term development strategies.
    Special Economic Zones or SEZs are designated areas with special economic regulations to attract investment.
    Public sector enterprises are owned by the government while private sector companies are privately owned.
    Liberalization Privatization and Globalization reforms were introduced in India in 1991.
    These reforms opened India's economy to foreign investment and reduced government control.
    The Green Revolution of the 1960s brought modern agricultural techniques to India increasing food production.
    The White Revolution increased milk production through the Operation Flood program.
    """, source="UPSC_economics")

    builder.get_stats()
    builder.save("train.txt")