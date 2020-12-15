package recfun

object RecFun extends RecFunInterface {

  def main(args: Array[String]): Unit = {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(s"${pascal(col, row)} ")
      println()
    }
  }

  /**
   * Exercise 1
   */
  def pascal(c: Int, r: Int): Int = {
    if (c == 0 || c == r) 1
    else pascal(c-1, r-1) + pascal(c, r-1)
  }

  /**
   * Exercise 2
   */
  def balance(chars: List[Char]): Boolean = {
    def balanceIter(chars: List[Char], leftCount: Int, rightCount: Int): Boolean = {
      // if the list is empty, check if left count equals to right count
      if(chars.isEmpty){
        if(leftCount == rightCount) true
        else false
      }
      // list is not empty, check if left count >= right count
      else if(leftCount < rightCount) false
      else if(chars.head == '('){
        balanceIter(chars.tail, leftCount + 1, rightCount)
      }else if(chars.head == ')'){
        balanceIter(chars.tail, leftCount, rightCount + 1)
      }else{
        balanceIter(chars.tail, leftCount, rightCount)
      }
    }
    balanceIter(chars, 0, 0)
  }

  /**
   * Exercise 3
   */
  def countChange(money: Int, coins: List[Int]): Int = {
    if (money > 0 && coins.nonEmpty)
      countChange(money - coins.head, coins) + countChange(money, coins.tail)
    else if (money == 0)
      1
    else
      0
  }
}
