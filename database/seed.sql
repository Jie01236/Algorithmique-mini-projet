USE socialmetrics;

INSERT INTO tweets (text, positive, negative) VALUES
('I love this product, it works perfectly', 1, 0),
('Amazing service and very friendly support', 1, 0),
('This update is excellent and useful', 1, 0),
('The experience was smooth and pleasant', 1, 0),
('I hate this app, it crashes all the time', 0, 1),
('Terrible support and very bad quality', 0, 1),
('This is disappointing and completely broken', 0, 1),
('The service is slow, expensive and useless', 0, 1),
('The interface is acceptable but not special', 0, 0),
('The announcement was neutral and factual', 0, 0),
('Great design but the performance is poor', 1, 1),
('Useful features, although the setup is painful', 1, 1),
('Excellent quality and fast delivery', 1, 0),
('I am happy with this reliable service', 1, 0),
('Perfect result, very useful and clean', 1, 0),
('Bad experience, I am angry and disappointed', 0, 1),
('Awful product, useless and frustrating', 0, 1),
('Worst update ever, nothing works', 0, 1),
('Bon service et tres bonne qualite', 1, 0),
('Mauvaise experience, service lent et horrible', 0, 1);
