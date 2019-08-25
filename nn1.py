import tensorflow as tf

#creates nodes in a graph
# "construction phase"

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)
print(result)

#define our session and launches the graph
sess = tf.Session()
#runs result
print(sess.run(result))

#output = sess.run(result)
#print(output)

#close session and release resources
sess.close()


#option utilizing with statement
#with tf.Session() as sess:
#    output = sess.run(result)
#    print(output)

#run on specific GPU
#with tf.Session() as sess:
#  with tf.device("/gpu:1"):
#    matrix1 = tf.constant([[3., 3.]])
#    matrix2 = tf.constant([[2.],[2.]])
#    product = tf.matmul(matrix1, matrix2)
