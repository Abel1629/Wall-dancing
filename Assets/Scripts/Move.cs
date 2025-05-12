using UnityEngine;

// it moves the walls, also it destroys them when colliding with a trigger

[RequireComponent(typeof(Rigidbody))]
public class Move : MonoBehaviour
{
    public float speed = 10.0f;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        Vector3 movement = new Vector3(0, 0, -speed) * Time.fixedDeltaTime;
        rb.MovePosition(rb.position + movement);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Destroy"))
        {
            Destroy(gameObject);
        }
    }
}
